import os.path
import sys
sys.path.append('./')
from typing import Tuple
import trimesh
from tqdm import tqdm
import torch
from torch import Tensor
from p_vae.pvae import SDFtoSDF
from utils import plot_march_fns as pmt_fns
from utils import sub_voxel_related_fns as pp_fns
from utils.positional_encoder_class import MYPositionalEncoder3D
# ----------------------------------------------------------------------------------------------------------------------------------------------------
from utils import encoder_decoder_loading as ed
from utils import encoder_related_fns as enc_fns
from utils.m_cube_fns import make_mcubes_from_voxels_ply_with_pad
from eval_fns import evaluate_all, write_evaluation_result
from training.train import CompletePartialScans as cp
from test_dataset import TESTLMDBOBJAVERSEPARTIALVIEWS

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class EvalObjaverse():
    def __init__(
            self,
            latent_dim: int,
            resolution: int,
            target_resolution: int,
            val_batch_size: int,
            test_lmdb_path: str,
            mesh_path: str,
            vae_checkpoint_path: str,
            common_obj_dir: str,
            views_dict_path: str,
            pre_trained: bool,
            image_resolution: int,
            ckpt_path: str,
            eval_dir: str,
            obj_dir: str,
            num_samples: int,
            num_views_for_test: int,
            min_range: int,
            max_range: int,

    ):
        super(EvalObjaverse, self).__init__()
        self.mesh_path = mesh_path
        self.common_obj_dir = common_obj_dir
        self.views_dict_path = views_dict_path
        self.image_resolution = image_resolution

        self.eval_dir = eval_dir
        self.obj_dir = obj_dir
        self.num_samples = num_samples
        self.num_views_for_test = num_views_for_test

        self.resolution = resolution
        self.target_resolution = target_resolution
        self.latent_dim = latent_dim

        self.val_batch_size = val_batch_size

        self.min_range = min_range
        self.max_range = max_range

        self.number_of_sub_voxels = 64

        number_of_sub_voxels = self.resolution // self.target_resolution
        self.number_of_sub_voxels = number_of_sub_voxels * number_of_sub_voxels * number_of_sub_voxels

        self.device = 'cuda:0'

        self.test_lmdb_path = test_lmdb_path

        if pre_trained:
            print("\n pre_trained: ", pre_trained)
            pre_trained_vae = SDFtoSDF.load_from_checkpoint(
                vae_checkpoint_path,
                map_location='cpu'
            )
            pre_trained_vae.freeze()
            pre_trained_vae.train(False)
            # del SDFtoSDF
            self.fdecoder = ed.load_decoder_from_checkpoint(pre_trained_vae, latent_dim)
            self.fencoder = ed.load_encoder_from_checkpoint(pre_trained_vae, latent_dim)

        self.penc_channels = 8 * self.latent_dim * 2
        self.positional_encoder_3d = MYPositionalEncoder3D(self.penc_channels)

        pretrained_model = cp.load_from_checkpoint(checkpoint_path=ckpt_path, map_location='cpu')

        pretrained_model.eval()
        pretrained_model.train(False)

        self.regular_transformer = pretrained_model.regular_transformer
        self.regular_transformer.eval()
        self.regular_transformer.train(False)

        self.mapping_down = pretrained_model.mapping_down
        self.mapping_down.eval()
        self.mapping_down.train(False)

        self.mapping_up = pretrained_model.mapping_up
        self.mapping_up.eval()
        self.mapping_up.train(False)

        self.conv1 = pretrained_model.conv1
        self.conv1.eval()
        self.conv1.train(False)
        self.conv2 = pretrained_model.conv2
        self.conv2.eval()
        self.conv2.train(False)
        self.conv3 = pretrained_model.conv3
        self.conv3.eval()
        self.conv3.train(False)

        self.regular_transformer = self.regular_transformer.to(device=self.device)
        self.mapping_down = self.mapping_down.to(device=self.device)
        self.mapping_up = self.mapping_up.to(device=self.device)
        self.conv1 = self.conv1.to(device=self.device)
        self.conv2 = self.conv2.to(device=self.device)
        self.conv3 = self.conv3.to(device=self.device)

        self.fencoder = self.fencoder.to(device=self.device)
        self.fdecoder = self.fdecoder.to(device=self.device)
        self.positional_encoder_3d = self.positional_encoder_3d.to(device=self.device)

        self.setup()

    def call_transformer_and_mapping_layers(self, transformer_input_sequence: torch.Tensor) -> torch.Tensor:
        transformer_input_sequence = transformer_input_sequence

        transformer_input_sequence_down = self.mapping_down(transformer_input_sequence)
        transformer_output_sequence = self.regular_transformer(transformer_input_sequence_down)
        transformer_output_sequence_up = self.mapping_up(transformer_output_sequence)

        return transformer_output_sequence_up

    def forward(self, sdf_latent_codes: torch.Tensor, uncertainty_latent_codes: torch.Tensor) -> torch.Tensor:
        batch_size = sdf_latent_codes.shape[0]

        concatenated_latent_codes = torch.cat((sdf_latent_codes, uncertainty_latent_codes), dim=2).to(device=sdf_latent_codes.device)
        concatenated_latent_codes_reshaped = concatenated_latent_codes.reshape([batch_size, self.number_of_sub_voxels, 2 * 8 * self.latent_dim])
        z_positionally_encoded_re = self.positional_encoder_3d(shape_of_positions=[batch_size, 4, 4, 4, self.penc_channels])
        assert z_positionally_encoded_re.shape == concatenated_latent_codes_reshaped.shape
        transformer_input_sequence = torch.cat((z_positionally_encoded_re, concatenated_latent_codes_reshaped), dim=2).to(device=z_positionally_encoded_re.device)
        transformer_output_sequence = self.call_transformer_and_mapping_layers(transformer_input_sequence)

        return transformer_output_sequence

    def encode_stuff(self, gt_sdf_sub_voxels: torch.Tensor,  gt_uncertainty_combined_voxel: torch.Tensor, train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = gt_sdf_sub_voxels.shape[0]
        # embed the sdfs -> sdf_latent_code
        sdf_latent_codes, sdf_std, sdf_var = enc_fns.prep_32cube_sub_voxels_and_encode(self.fencoder, gt_sdf_sub_voxels, self.number_of_sub_voxels, self.latent_dim, self.target_resolution, train)

        # embed the uncertainty values -> uncertainty_latent_code using a conv3D with stride to shrink its dimensionality
        # adding a batch dim first
        gt_uncertainty_combined_voxel_batched = gt_uncertainty_combined_voxel.unsqueeze(1)
        assert (gt_uncertainty_combined_voxel_batched.shape == (batch_size, 1, self.resolution, self.resolution, self.resolution))

        uncertainty_latent_codes = self.conv1(gt_uncertainty_combined_voxel_batched)
        uncertainty_latent_codes = self.conv2(uncertainty_latent_codes)
        uncertainty_latent_codes = self.conv3(uncertainty_latent_codes)

        # [64,16,16,16]
        uncertainty_latent_codes = uncertainty_latent_codes.reshape([-1, 64, 4, 4, 4, 4, 4, 4])
        # 4x4x4 xyz patches to the front, channels to back
        # Batch, patch xyz, within-patch xyz, channels
        uncertainty_latent_codes = uncertainty_latent_codes.permute([0, 2, 4, 6, 3, 5, 7, 1])  # to put spatial dims upfront to align with sdf for concatenation

        uncertainty_latent_codes_r = uncertainty_latent_codes.reshape([batch_size, self.number_of_sub_voxels, self.latent_dim, 2, 2, 2])
        return (sdf_latent_codes, uncertainty_latent_codes_r)

    def fwd(self, batch: list, test: bool) -> tuple[Tensor, Tensor, Tensor]:
        (object_indices, mesh_file_name, mesh_name, folder_name, combined_sdf_voxel, combined_uncertainty_voxel_normalized) = batch
        batch_size = object_indices.shape[0]
        # -----------------------------------------------------------------------------------------------------------------------------------------
        combined_sdf_sub_voxels = pp_fns.sub_divide_gt_and_normalize(combined_sdf_voxel, self.number_of_sub_voxels, self.target_resolution)
        # uncertainty values are normalized to [-1,1] already in dataset class
        sdf_latent_codes, uncertainty_latent_codes = self.encode_stuff(combined_sdf_sub_voxels, combined_uncertainty_voxel_normalized, False)
        transformer_output_sequence_up = self.forward(sdf_latent_codes, uncertainty_latent_codes)

        return (sdf_latent_codes, uncertainty_latent_codes, transformer_output_sequence_up)

    def export_gt(self, data_to_export):
        gt_vertices = data_to_export["gt_vertices"]
        gt_faces = data_to_export["gt_faces"]

        object_index = data_to_export["object_index"]
        folder_name = data_to_export["folder_name"]
        mesh_name = data_to_export["mesh_name"]
        mesh_name_ = mesh_name.rsplit('_')[0]
        gt_dir = os.path.join(self.common_obj_dir, 'GT', "obj_dir")
        gt_mesh_file_name = os.path.join(gt_dir, "GT" + "_" + folder_name + "_" + mesh_name_ + "_ObjID=" + str(object_index) + ".ply")
        gt_mesh = trimesh.Trimesh(vertices=gt_vertices, faces=gt_faces)
        gt_mesh.export(gt_mesh_file_name)
        return gt_mesh_file_name

    def test(self) -> None:

        print("\n min_range", self.min_range, "-- max_range:", self.max_range)

        for i in tqdm(range(self.min_range, self.max_range, 1), "Objaverse Test Samples:"):
            batch = self.test_dataset[i]

            (
                object_indices,
                mesh_file_name, mesh_name, folder_name, combined_sdf_voxel, combined_uncertainty_voxel_normalized, gt_vertices, gt_faces, gt_sdf_latent_codes
            ) = batch
            # add batch
            object_indices = torch.tensor([object_indices])
            combined_sdf_voxel = combined_sdf_voxel.unsqueeze(0).to(device=self.device)
            combined_uncertainty_voxel_normalized = combined_uncertainty_voxel_normalized.unsqueeze(0).to(device=self.device)
            gt_sdf_latent_codes = torch.from_numpy(gt_sdf_latent_codes).unsqueeze(0).to(device=self.device)
            batch_size = object_indices.shape[0]
            assert object_indices.shape == (self.val_batch_size,)
            test_batch = [
                object_indices,
                mesh_file_name, mesh_name, folder_name, combined_sdf_voxel, combined_uncertainty_voxel_normalized
            ]

            stuff = self.fwd(test_batch, test=True)

            sdf_latent_codes, uncertainty_latent_codes, transformer_output_sequence_up = stuff

            # just for vis once only
            if False:
                # once, write all the GT(actual dataset) as ply file
                data_to_export: dict = {"gt_vertices": gt_vertices.cpu().numpy(), "gt_faces": gt_faces.cpu().numpy(), "object_index": object_indices.detach().cpu().item(), "mesh_name": mesh_name, "folder_name": folder_name}
                gt_export_file_name = self.export_gt(data_to_export)

            transformer_output_sequence_up_reshaped = transformer_output_sequence_up.reshape(batch_size, self.number_of_sub_voxels, self.latent_dim, 2, 2, 2)

            # visualization and
            dict_args_vis = {
                "Transformer": transformer_output_sequence_up_reshaped,
                "GT_SDF": gt_sdf_latent_codes,  # constant, what is written in LMDB

            }

            dict_args_variables = {
                "number_of_sub_voxels": self.number_of_sub_voxels,
                "latent_dim": self.latent_dim,
                "target_resolution": self.target_resolution,
                "resolution": self.resolution,
                "batch_size": batch_size,
                "hausdorff_scale": 1,
                "chamfer_scale": 1,
            }
            # this is decoded now
            dict_data_vis = pmt_fns.generate_any_data_for_plotting(dict_args_vis, dict_args_variables, self.fdecoder)

            dict_args_eval: dict = {}
            selected_index = object_indices.detach().item()
            collected_data_dict_for_plotting = pmt_fns.collect_any_generated_data_for_plotting(dict_data_vis, batch_idx=0)
            keys = [key for key in collected_data_dict_for_plotting.keys()]
            for k in range(len(keys)):
                current_key = keys[k]
                collected_data_current = collected_data_dict_for_plotting.get(current_key)
                if current_key.startswith("Transformer") or current_key.startswith("Transformer_filtered"):
                    export_file_name = os.path.join(self.obj_dir, str(current_key) + "_" + folder_name + "_" + mesh_name + "_ObjID=" + str(selected_index) + ".ply")
                    make_mcubes_from_voxels_ply_with_pad(collected_data_current, export_file_name)
                if current_key.startswith("Transformer") and export_file_name.endswith(".ply"):
                    dict_args_eval["Transformer_file"] = export_file_name

            # ##### TODO , If you do not need it, comment this section!
            # CLIP Input , for vis only:
            # we need to un-normalized uncertainty first
            un_normalized_uncertainty_voxel = (combined_uncertainty_voxel_normalized.squeeze(0) * 50) + 50
            # This is only for visualization purpose
            clipped_combined_sdf_voxel = combined_sdf_voxel.squeeze(0)-torch.clip(un_normalized_uncertainty_voxel-8, min=0.0)*0.003
            input_dir = os.path.join(self.common_obj_dir, "Inputs", "Input" + "_num_views-" + str(self.num_views_for_test))
            if not os.path.isdir(input_dir):
                os.mkdir(input_dir)
            input_export_file_name = os.path.join(input_dir, "Input" + "_" + folder_name + "_" + mesh_name.rsplit('_')[0] + "_ObjID=" + str(selected_index) + ".ply")
            if not os.path.isfile(input_export_file_name):
                make_mcubes_from_voxels_ply_with_pad(clipped_combined_sdf_voxel.cpu().numpy(), input_export_file_name)
            dict_args_eval["Input_file"] = input_export_file_name
            # #####

            #  we here use the actual dataset GT and not the marched gt_sdf_voxel as GT for the mesh-level-metrics
            gt_dir = os.path.join(self.common_obj_dir, 'GT', "obj_dir")
            gt_export_file_name = os.path.join(gt_dir, "GT" + "_" + folder_name + "_" + mesh_name.rsplit('_')[0] + "_ObjID=" + str(selected_index) + ".ply")
            dict_args_eval["GT_file"] = gt_export_file_name

            # Decoded ones on voxel level
            dict_args_eval["Completed_voxel"] = collected_data_dict_for_plotting["Transformer"]
            # Here to evaluate voxel-level metrics, we have no choice but taking the marched GT_sdf_voxel
            dict_args_eval["GT_voxel"] = collected_data_dict_for_plotting["GT_SDF"]
            # dict_args_eval["Input_voxel"] = collected_data_dict_for_plotting["Input"]
            dict_args_eval["Object_index"] = object_indices[0]
            dict_args_eval["num_samples"] = self.num_samples

            # now evaluate this object:
            eval_results = evaluate_all(dict_args_eval, dict_args_variables)
            write_evaluation_result(eval_results, self.eval_dir)

    def setup(self):

        self.test_dataset = TESTLMDBOBJAVERSEPARTIALVIEWS(self.mesh_path, self.views_dict_path,  self.test_lmdb_path, self.obj_dir,
                                                     self.image_resolution, self.num_views_for_test, self.resolution, device=self.device)

        print("\n test_dataset len:", len(self.test_dataset))

