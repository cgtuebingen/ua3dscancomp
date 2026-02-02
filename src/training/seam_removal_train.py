import sys
from typing import Tuple, Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
sys.path.append('/home/zakeri/Documents/Codes/MyCodes/Proposal2/ua3dscancomp-gitbub/src/')
import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from train_dataset import LMDBOBJAVERSEPARTIALVIEWS
from p_vae.pvae import SDFtoSDF
from utils import transformer_visualizations as tv

import loss_helper_fns as l_fn
from utils import plot_march_fns as pmt_fns
from utils import sub_voxel_related_fns as pp_fns
from utils.positional_encoder_class import MYPositionalEncoder3D
# ----------------------------------------------------------------------------------------------------------------------------------------------------
from utils import encoder_decoder_loading as ed
from utils.helper_fns import concatenate_for_given_dim

# from transformers.optimization import get_cosine_schedule_with_warmup
from utils import encoder_related_fns as enc_fns
from utils.m_cube_fns import make_mcubes_from_voxels_obj_with_pad

from subvolume_devision import (collect_sub_voxels_to_voxel_with_batch, subvdivide_voxel_with_batch)
from seam_removal_model import SeamRemoval
# --------------------------------------------------------------------------------------------------------------------------------------------------------
class RemoveSeams(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        resolution: int,
        target_resolution: int,
        batch_size: int,
        val_batch_size: int,
        learning_rate: float,
        warmup_ratio: float,
        train_lmdb_path: str,
        val_lmdb_path: str,
        test_lmdb_path: str,
        mesh_path: str,
        value_range: int,
        vae_checkpoint_path: str,
        marching_cube_result_dir: str,
        layers: int,
        dim_size: int,
        heads: int,
        pre_trained: bool,
        image_resolution: int,
        resume_on_previous_model: bool,
        previous_model_ckpt_path: str,
        eval_dir: str,
        obj_dir: str,
        num_samples: int,
        num_views_for_test: int,
        first_transformer_ckpt: str,

    ):
        super(RemoveSeams, self).__init__()
        self.save_hyperparameters()

        self.number_of_sub_voxels = 64
        self.l1_loss = nn.L1Loss(reduction="mean")

        number_of_sub_voxels = self.hparams.resolution // self.hparams.target_resolution
        self.number_of_sub_voxels = number_of_sub_voxels * number_of_sub_voxels * number_of_sub_voxels

        # self.my_selected_indices = [i for i in range(0, 100, 5)]  # this script
        self.my_selected_indices = [44, 20, 75, 45, 30, 10, 1, 8, 16, 32, 64, 128, 256, 512]

        if self.hparams.pre_trained:
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

        self.penc_channels = 8 * self.hparams.latent_dim * 2
        self.positional_encoder_3d = MYPositionalEncoder3D(self.penc_channels)
        if pre_trained:
            from Partial3DScan.Developement.experiments.fulldataset.bf16_mixed.cvpr_submission.completePartialScans import CompletePartialScans as sps
            pretrained_model = sps.load_from_checkpoint(checkpoint_path=first_transformer_ckpt, map_location='cpu')

            pretrained_model.freeze()
            pretrained_model.train(False)

            self.regular_transformer = pretrained_model.regular_transformer
            self.regular_transformer.train(False)

            self.mapping_down = pretrained_model.mapping_down
            self.mapping_down.train(False)

            self.mapping_up = pretrained_model.mapping_up
            self.mapping_up.train(False)

            self.conv1 = pretrained_model.conv1
            self.conv1.train(False)
            self.conv2 = pretrained_model.conv2

            self.conv2.train(False)
            self.conv3 = pretrained_model.conv3
            self.conv3.train(False)

        self.sr = SeamRemoval()
        # self.seam_removal_conv3d_1 = torch.nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), padding_mode='replicate')
        # # self.seam_removal_conv3d_2 = torch.nn.Conv3d(in_channels=4, out_channels=2, kernel_size=(9, 9, 9), stride=(1, 1, 1), padding=(4, 4, 4), padding_mode='replicate')
        # self.seam_removal_conv3d_3 = torch.nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), padding_mode='replicate')
        # self.seam_removal_conv3d_4 = torch.nn.Conv3d(in_channels=8, out_channels=1, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2), padding_mode='replicate')
    def call_seam_removal_with_conv3d(self, transformer_output_sequence_up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = transformer_output_sequence_up.shape[0]
        transformer_output_sequence_up_reshaped = transformer_output_sequence_up.reshape(batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2)
        with torch.no_grad():
            decoded_trans_output = self.fdecoder(transformer_output_sequence_up_reshaped)
        decoded_trans_output_reshaped = decoded_trans_output.reshape([batch_size, self.number_of_sub_voxels, self.hparams.target_resolution, self.hparams.target_resolution, self.hparams.target_resolution])
        # decoded_trans_output_perm = decoded_trans_output_perm.reshape([batch_size, self.number_of_sub_voxels, self.hparams.target_resol])
        decoded_trans_output_collected = collect_sub_voxels_to_voxel_with_batch(decoded_trans_output_reshaped, self.hparams.resolution)  # [B, 128, 128, 128]
        del decoded_trans_output_reshaped
        # decoded_trans_output_reshaped_sub_voxels = subvdivide_voxel_with_batch(decoded_trans_output_reshaped, self.hparams.target_resolution)
        # reshaping from [B, 64, 4096] to [B, 4096, 4, 4, 4]
        decoded_trans_output_collected_u = decoded_trans_output_collected.unsqueeze(1)
        # del decoded_trans_output_collected
        x = decoded_trans_output_collected_u.detach()  # don't backprop across here
        x = self.sr.forward((x))
        # x = self.seam_removal_conv3d_1(x)
        # x = torch.relu(x)
        # # x = self.seam_removal_conv3d_2(x)
        # # x = torch.relu(x)
        # x = self.seam_removal_conv3d_3(x)
        # x = torch.relu(x)
        # x = self.seam_removal_conv3d_4(x)
        # x = x.squeeze(1)
        return (x, decoded_trans_output_collected)
    def call_transformer_and_mapping_layers(self, transformer_input_sequence: torch.Tensor) -> torch.Tensor:
        transformer_input_sequence = transformer_input_sequence

        transformer_input_sequence_down = self.mapping_down(transformer_input_sequence)
        #with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        transformer_output_sequence = self.regular_transformer(transformer_input_sequence_down)
        transformer_output_sequence_up = self.mapping_up(transformer_output_sequence)

        return transformer_output_sequence_up

    def forward(self, sdf_latent_codes: torch.Tensor, uncertainty_latent_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # This part is the same for training and validation
            batch_size = sdf_latent_codes.shape[0]
            # concatenate latent codes of sdf and uncertainty values-------------------------------------------------------------------
            concatenated_latent_codes = concatenate_for_given_dim(sdf_latent_codes, uncertainty_latent_codes, cat_dim=2)
            concatenated_latent_codes_reshaped = concatenated_latent_codes.reshape([batch_size, self.number_of_sub_voxels, 2 * 8 * self.hparams.latent_dim])

            # Positional Embeder-------------------------------------------------------------------------------------------------------
            z_positionally_encoded_re = self.positional_encoder_3d(shape_of_positions=[batch_size, 4, 4, 4, self.penc_channels])
            # Adding latent code with positional embedding----------------------------------------------------------------------------
            assert z_positionally_encoded_re.shape == concatenated_latent_codes_reshaped.shape
            # CAT---------------------------------------------------------------------------------------------------------------------
            transformer_input_sequence = concatenate_for_given_dim(z_positionally_encoded_re, concatenated_latent_codes_reshaped, cat_dim=2)
            # # Transformer ----------------------------------------------------------------------------------------------------------
            # MLP --------------------------------------------------------------------------------------------------------------------

            transformer_output_sequence = self.call_transformer_and_mapping_layers(transformer_input_sequence)

        del concatenated_latent_codes, concatenated_latent_codes_reshaped, z_positionally_encoded_re, transformer_input_sequence

        conv3d_output, decoded_trans_output_collected = self.call_seam_removal_with_conv3d(transformer_output_sequence)
        return (conv3d_output, decoded_trans_output_collected)

    def encode_stuff(self, gt_sdf_sub_voxels: torch.Tensor,  gt_uncertainty_combined_voxel: torch.Tensor, train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = gt_sdf_sub_voxels.shape[0]
        # embed the sdfs -> sdf_latent_code
        sdf_latent_codes, sdf_std, sdf_var = enc_fns.prep_32cube_sub_voxels_and_encode(self.fencoder, gt_sdf_sub_voxels, self.number_of_sub_voxels, self.hparams.latent_dim, self.hparams.target_resolution, train)

        # embed the uncertainty values -> uncertainty_latent_code using a conv3D with stride to shrink its dimensionality
        # adding a batch dim first
        gt_uncertainty_combined_voxel_batched = gt_uncertainty_combined_voxel.unsqueeze(1)
        assert (gt_uncertainty_combined_voxel_batched.shape == (batch_size, 1, self.hparams.resolution, self.hparams.resolution, self.hparams.resolution))
        with torch.no_grad():
            uncertainty_latent_codes = self.conv1(gt_uncertainty_combined_voxel_batched)
            uncertainty_latent_codes = self.conv2(uncertainty_latent_codes)
            uncertainty_latent_codes = self.conv3(uncertainty_latent_codes)

        # TODO : I am changed here
        # [64,16,16,16]
        uncertainty_latent_codes = uncertainty_latent_codes.reshape([-1, 64, 4, 4, 4, 4, 4, 4])
        # 4x4x4 xyz patches to the front, channels to back
        # Batch, patch xyz, within-patch xyz, channels
        uncertainty_latent_codes = uncertainty_latent_codes.permute([0, 2, 4, 6, 3, 5, 7, 1])  # to put spatial dims upfront to align with sdf for concatenation

        uncertainty_latent_codes_r = uncertainty_latent_codes.reshape([batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2])
        return (sdf_latent_codes, uncertainty_latent_codes_r)

    # def slice_loss_transition_slices(self, conv3d_output: torch.Tensor):
    #     # conv3d_output = loss_data["conv3d_output"]  # [128, 128, 128]
    #     # decoded_gt_collected = loss_data["decoded_gt_collected"]  # [128, 128, 128]
    #     left_x = conv3d_output[:, [31, 63, 95], :, :]
    #     right_x = conv3d_output[:, [32, 64, 96], :, :]
    #
    #     left_y = conv3d_output[:, :, [31, 63, 95], :]
    #     right_y = conv3d_output[:, :, [32, 64, 96], :]
    #
    #     left_z = conv3d_output[:, :, :, [31, 63, 95]]
    #     right_z = conv3d_output[:, :, :, [32, 64, 96]]
    #
    #     return torch.mean(torch.abs(right_x-left_x))+torch.mean(torch.abs(right_y-left_y))+torch.mean(torch.abs(right_z-left_z))
    #
    # def slice_loss(self, conv3d_output: torch.Tensor):
    #     left_x = conv3d_output[:, :-1, :, :]
    #     right_x = conv3d_output[:, 1:, :, :]
    #
    #     left_y = conv3d_output[:, :, :-1, :]
    #     right_y = conv3d_output[:, :, 1:, :]
    #
    #     left_z = conv3d_output[:, :, :, :-1]
    #     right_z = conv3d_output[:, :, :, 1:]
    #
    #     return torch.mean(torch.abs(right_x-left_x))+torch.mean(torch.abs(right_y-left_y))+torch.mean(torch.abs(right_z-left_z))

    def sdf_grad_loss(self, conv3d_output: torch.Tensor):
        left_x  = conv3d_output[:, :-2, :-2, :-2]
        right_x = conv3d_output[:,  2:, :-2, :-2]

        left_y  = conv3d_output[:, :-2, :-2, :-2]
        right_y = conv3d_output[:, :-2,  2:, :-2]

        left_z  = conv3d_output[:, :-2, :-2, :-2]
        right_z = conv3d_output[:, :-2, :-2,  2:]

        grad_x = right_x - left_x
        grad_y = right_y - left_y
        grad_z = right_z - left_z

        central_differences_scale = 128/4  # inverse of the size of 2 voxels
        # normalize
        grad_x *= central_differences_scale
        grad_y *= central_differences_scale
        grad_z *= central_differences_scale

        jacobian = torch.stack([grad_x, grad_y, grad_z], dim=1)  # [B, 3, 126, 126, 126]

        grad_norm = torch.linalg.vector_norm(jacobian, dim=1)

        # eikonal property
        eikonal_loss = torch.mean(torch.square(grad_norm-1))

        # left_grad_x  = grad_x[:, :-1, :-1, :-1]
        # right_grad_x = grad_x[:,  1:, :-1, :-1]
        # 
        # left_grad_y  = grad_y[:, :-1, :-1, :-1]
        # right_grad_y = grad_y[:, :-1,  1:, :-1]
        # 
        # left_grad_z  = grad_z[:, :-1, :-1, :-1]
        # right_grad_z = grad_z[:, :-1, :-1,  1:]
        # 
        # # normalize the grad
        # diff_grad_x = right_grad_x - left_grad_x
        # diff_grad_y = right_grad_y - left_grad_y
        # diff_grad_z = right_grad_z - left_grad_z
        # 
        # # normalize
        # diff_grad_x *= scale
        # diff_grad_y *= scale
        # diff_grad_z *= scale
        # 
        # laplace_norm = torch.square(diff_grad_x)+torch.square(diff_grad_y)+torch.square(diff_grad_z)
        # 
        # laplace_loss = torch.mean(laplace_norm)

        return eikonal_loss #, laplace_loss

    def fwd(self, batch: list, train: bool, val: bool, test: bool) -> tuple[dict[str, Any], tuple[Tensor, Tensor, Tensor, Tensor]]:
        with torch.no_grad():
            (object_indices, mesh_file_name, mesh_name, folder_name, combined_sdf_voxel, combined_uncertainty_voxel, gt_sdf_latent_codes) = batch
            batch_size = object_indices.shape[0]
            del batch
            # -----------------------------------------------------------------------------------------------------------------------------------------
            combined_sdf_sub_voxels = pp_fns.sub_divide_gt_and_normalize(combined_sdf_voxel, self.number_of_sub_voxels, self.hparams.target_resolution)
            del combined_sdf_voxel
            # uncertainty values are normalized to [-1,1] already in dataset class
            sdf_latent_codes, uncertainty_latent_codes = self.encode_stuff(combined_sdf_sub_voxels, combined_uncertainty_voxel, False)
            del combined_sdf_sub_voxels, combined_uncertainty_voxel
            # FORWARD CAlL---------------------------------------------------------------------------------------------------------------------------------
            # if I want this to run , my val_batch and my train_batch need to be the same.
        conv3d_output, decoded_trans_output_collected = self.forward(sdf_latent_codes, uncertainty_latent_codes)

        eikonal_loss = self.sdf_grad_loss(conv3d_output)

        # slice_loss_transition_slices = self.slice_loss_transition_slices(conv3d_output)

        consistency_loss = torch.mean(torch.abs(conv3d_output-decoded_trans_output_collected))

        # consistency_loss_out = torch.mean(torch.relu(conv3d_output - decoded_trans_output_collected))
        # consistency_loss_in = torch.mean(torch.relu(decoded_trans_output_collected - conv3d_output))

        # for loss calculation against the gt_sdf_latent_code that is precalculated from LMDB
        # decode the GT

        # loss_l1 = grad_norm_loss + smooth_grad_loss * 0.001 + consistency_loss * 0.01  #consistency_loss_out * 0.05 + consistency_loss_in * 0.1 # old implementation no normalization

        # # v49
        # eikonal_loss = eikonal_loss * 5.0e-4
        # laplace_loss = laplace_loss * 1.0e-10
        # consistency_loss = consistency_loss

        # v50
        # eikonal_loss = eikonal_loss * 1.0e-3
        # laplace_loss = laplace_loss * 1.0e-6
        # consistency_loss = consistency_loss

        # # v52
        # eikonal_loss = eikonal_loss * 1.0e-1
        # laplace_loss = laplace_loss * 1.0e-4
        # consistency_loss = consistency_loss

        # v55
        eikonal_loss = eikonal_loss * 5.0e-2
        # laplace_loss = laplace_loss * 1.0e-6
        consistency_loss = consistency_loss

        # loss_l1 = eikonal_loss + laplace_loss + consistency_loss

        # v56
        loss_l1 = eikonal_loss + consistency_loss

        # with torch.no_grad():
        #     decoded_gt = self.fdecoder(gt_sdf_latent_codes)
        #     decoded_gt_reshaped = decoded_gt.reshape([batch_size, self.number_of_sub_voxels, self.hparams.target_resolution, self.hparams.target_resolution, self.hparams.target_resolution])
        #     decoded_gt_collected = collect_sub_voxels_to_voxel_with_batch(decoded_gt_reshaped, self.hparams.resolution)

        # calculate losses:
        # loss_l1 = self.l1_loss(conv3d_output, decoded_gt_collected)
        # log losses
        loss_dict = {"loss": loss_l1, "eikonal_loss": eikonal_loss, "consistency_loss": consistency_loss}

        if train:
            # loss_l1 = self.l1_loss(conv3d_output, decoded_gt)
            # loss_dict = {"l1_loss": loss_l1}
            loss_log = l_fn.create_log_losses_for_given_dict(loss_dict, stage="training")
            self.log_dict(loss_log, batch_size=self.hparams.batch_size, sync_dist=True)
            self.log("train_loss", loss_l1, batch_size=self.hparams.batch_size, sync_dist=True)
            # print("\n training loss:", loss_l1)

        elif val:
            loss_log = l_fn.create_log_losses_for_given_dict(loss_dict, stage="val")
            self.log_dict(loss_log, batch_size=self.hparams.val_batch_size, sync_dist=True)
            self.log("val_loss", loss_l1, batch_size=self.hparams.val_batch_size, sync_dist=True)

        elif test:
            pass

        else:
            raise ("\ninvalid stage!")

        return (loss_dict, (sdf_latent_codes, uncertainty_latent_codes, decoded_trans_output_collected, conv3d_output))

    def training_step(self, batch: list, batch_idx: int) -> dict:
        # (
        #     object_indices,
        #     mesh_file_name, mesh_name, folder_name, combined_sdf_voxel, combined_uncertainty_voxel, gt_sdf_latent_codes
        # ) = batch
        loss_dict, stuff = self.fwd(batch,  train=True, val=False, test=False)

        # sdf_latent_codes, uncertainty_latent_codes, transformer_output_sequence_up = stuff
        return loss_dict

    def validation_step(self, batch: list, batch_idx: int) -> None:

        self.fdecoder.eval()
        self.fencoder.eval()

        assert not self.fdecoder.batchNorm3d5.track_running_stats
        assert not self.fdecoder.batchNorm3d6.track_running_stats
        assert not self.fdecoder.batchnorm3d7.track_running_stats
        assert not self.fdecoder.training

        (
            object_indices,
            mesh_file_name, mesh_name, folder_name, combined_sdf_voxel, combined_uncertainty_voxel, gt_sdf_latent_codes
        ) = batch

        batch_size = object_indices.shape[0]
        assert object_indices.shape == (self.hparams.val_batch_size,)
        loss_dict, stuff = self.fwd(batch, train=False, val=True, test=False)

        sdf_latent_codes, uncertainty_latent_codes, decoded_trans_output_collected, conv3d_output = stuff

        # just for vis
        # transformer_output_sequence_up_reshaped = transformer_output_sequence_up.reshape(batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2)
        # visualization and tensorboard---------------------------
        dict_arguments_for_vis = {
            "InputSdfLatentCodes": sdf_latent_codes,
            # "uncertainty_latent_codes": uncertainty_latent_codes,
            # "TransformerOutput": transformer_output_sequence_up_reshaped,
            "GT_sdfLatentCodes": gt_sdf_latent_codes,


        }

        dict_arguments_of_variables = {
            "number_of_sub_voxels": self.number_of_sub_voxels,
            "latent_dim": self.hparams.latent_dim,
            "target_resolution": self.hparams.target_resolution,
            "resolution": self.hparams.resolution,
            "batch_size": batch_size,
            "Uncertainty": combined_uncertainty_voxel,
            "conv3d_output": conv3d_output,
            "decoded_trans_output_collected": decoded_trans_output_collected,
        }

        self.plot_march_and_login_tensorboard(dict_arguments_for_vis, dict_arguments_of_variables, object_indices, batch_size)

    def test_step(self, batch: list) -> None:
        (
            object_indices,
            mesh_file_name, mesh_name, folder_name, combined_sdf_voxel, combined_uncertainty_voxel, gt_sdf_latent_codes
        ) = batch
        batch_size = object_indices.shape[0]
        assert object_indices.shape == (self.hparams.val_batch_size,)

        loss_dict, stuff = self.fwd(batch, train=False, val=False, test=True)
        sdf_latent_codes, uncertainty_latent_codes, transformer_output_sequence_up = stuff
        # I want them to be marched, so I get to the results.

        # just for vis
        transformer_output_sequence_up_reshaped = transformer_output_sequence_up.reshape(batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2)
        # sdf_latent_codes = sdf_latent_codes.to(dtype=torch.float32)
        # sdf_latent_codes = transformer_output_sequence_up_reshaped.to(dtype=torch.float32)
        # gt_sdf_latent_codes = gt_sdf_latent_codes.to(dtype=torch.float32)
        # visualization and tensorboard---------------------------
        dict_args_vis = {
            "InputSdfLatentCodes": sdf_latent_codes,
            "TransformerOutput": transformer_output_sequence_up_reshaped,
            "GT_sdfLatentCodes": gt_sdf_latent_codes,  # constant, what is written in LMDB


        }

        dict_args_variables = {
            "number_of_sub_voxels": self.number_of_sub_voxels,
            "latent_dim": self.hparams.latent_dim,
            "target_resolution": self.hparams.target_resolution,
            "resolution": self.hparams.resolution,
            "batch_size": batch_size,
        }
        # this is decoded now
        dict_data_vis = pmt_fns.generate_any_data_for_plotting(dict_args_vis, dict_args_variables, self.fdecoder)
        dict_args_eval: dict = {}
        for b in range(batch_size):
            selected_index = object_indices.detach().item()
            # if selected_index in self.my_selected_indices:
            collected_data_dict_for_plotting = pmt_fns.collect_any_generated_data_for_plotting(dict_data_vis, batch_idx=b)
            keys = [key for key in collected_data_dict_for_plotting.keys()]
            for i in range(len(keys)):
                current_key = keys[i]
                collected_data_current = collected_data_dict_for_plotting.get(current_key)
                # current_epoch = self.trainer.current_epoch
                current_name = str(current_key)  # + "_" + "epoch-" + str(current_epoch)
                export_file_name_obj = self.hparams.obj_dir + str(current_key) + '-' + 'ObjID=' + str(selected_index) + '_' + '.obj'
                make_mcubes_from_voxels_obj_with_pad(collected_data_current, export_file_name_obj)
                print()
                if export_file_name_obj.startswith("Input") and export_file_name_obj.endswith(".obj"):
                    dict_args_eval["Input"] = export_file_name_obj
                if export_file_name_obj.startswith("Transformer") and export_file_name_obj.endswith(".obj"):
                    dict_args_eval["Transformer"] = export_file_name_obj
                if export_file_name_obj.startswith("GT") and export_file_name_obj.endswith(".obj"):
                    dict_args_eval["GT"] = export_file_name_obj
                # now evaluate this object:
                dict_args_eval["Completed_voxel"] = transformer_output_sequence_up_reshaped[b]
                dict_args_eval["GT_voxel"] = gt_sdf_latent_codes[b]
                dict_args_eval["Input_voxel"] = sdf_latent_codes[b]
                dict_args_eval["Object_index"] = object_indices[b]
                dict_args_eval["Object_index"] = self.hparams.num_samples

    def plot_march_and_login_tensorboard(self, dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, object_indices: torch.Tensor, batch_size: int) -> None:
        # this is decoded now
        data_dict_for_vis = pmt_fns.generate_any_data_for_plotting(dict_arguments_for_vis, dict_arguments_of_variables, self.fdecoder)
        # Diff------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        diff_decoded_trans_output_vs_conv3d_output = torch.subtract(
            dict_arguments_of_variables["conv3d_output"],
            dict_arguments_of_variables["decoded_trans_output_collected"]
        )
        # # diff_transformer_output_vs_latent_codes = diff_transformer_output_vs_latent_codes.to(dtype=torch.float32)
        data_dict_for_vis["diff_decoded_trans_output_vs_conv3d_output"] = diff_decoded_trans_output_vs_conv3d_output

        data_dict_for_vis["conv3d_output"] = dict_arguments_of_variables["conv3d_output"]
        data_dict_for_vis["decoded_trans_output_collected"] = dict_arguments_of_variables["decoded_trans_output_collected"]

        for b in range(batch_size):
            selected_index = object_indices[b].detach().cpu().item()
            if selected_index in self.my_selected_indices:
                collected_data_dict_for_plotting = pmt_fns.collect_any_generated_data_for_plotting(data_dict_for_vis, batch_idx=b)
                collected_data_dict_for_plotting["Uncertainty"] = dict_arguments_of_variables["Uncertainty"][b].squeeze().cpu().numpy()
                plots = tv.generate_plot_for_given_dict_of_items(collected_data_dict_for_plotting, self.hparams.resolution, number_of_slices=2, plot_scale_factor=2, plot_range=1)
                self.login_to_tensorboard(plots, selected_index, number_of_slices=2)
                pmt_fns.march_any_results_every_n_epoch(collected_data_dict_for_plotting, selected_index, self.current_epoch, self.trainer.global_step, self.hparams.marching_cube_result_dir)

    def login_to_tensorboard(self, plots: list, selected_index: int, number_of_slices: int):
        # plot and log slices
        for sl in range(0, number_of_slices, 1):
            # plot everything
            plot = tv.plot_for_given_dict_of_items(sl, plots, given_device=self.device)
            plot = plot.to(self.device)
            # show in tensorboard
            self.logger.experiment.add_image(
                "mesh-Id-{}_slice-{}".format(selected_index, sl),
                plot,
                self.global_step,
            )

    def setup(self, stage: str) -> None:

        self.train_dataset = LMDBOBJAVERSEPARTIALVIEWS(self.hparams.mesh_path, self.hparams.train_lmdb_path, self.hparams.marching_cube_result_dir,
                                                       self.hparams.image_resolution, self.hparams.resolution, device=self.device)

        print("\n train_dataset len:", len(self.train_dataset))

        # for validation while training

        self.val_dataset = LMDBOBJAVERSEPARTIALVIEWS(self.hparams.mesh_path, self.hparams.val_lmdb_path, self.hparams.marching_cube_result_dir,
                                                     self.hparams.image_resolution, self.hparams.resolution, device=self.device)
        self.val_dataset.len = 200
        print("\n val_dataset len:", len(self.val_dataset))

        # for evaluation
        # print("\n test dataset num_views:", self.trainer.model.num_views_for_test)
        self.test_dataset = LMDBOBJAVERSEPARTIALVIEWS(self.hparams.mesh_path, self.hparams.test_lmdb_path, self.hparams.marching_cube_result_dir,
                                                      self.hparams.image_resolution, self.hparams.resolution, device=self.device)

        # self.test_dataset.len = 100
        print("\n test_dataset len:", len(self.test_dataset))

        torch.multiprocessing.set_start_method('spawn')

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,

        )
        return dataloader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            # persistent_workers=True,
            drop_last=True,

        )
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,

        )

    def configure_optimizers(self):
        # we exclude fdecoer params from optimizer because it fucks them up, yes you heard me!
        dont_train_those = []
        for k, _ in self.fdecoder.named_parameters():
            dont_train_those.append(k)

        # we exclude fencoder params from optimizer because it fucks them up, yes you heard me!
        dont_train_those = []
        for k, _ in self.fencoder.named_parameters():
            dont_train_those.append(k)

        # print(params)
        dont_train_those = []
        for k, _ in self.regular_transformer.named_parameters():
            dont_train_those.append(k)

        # print(params)
        dont_train_those = []
        for k, _ in self.conv1.named_parameters():
            dont_train_those.append(k)

        # print(params)
        dont_train_those = []
        for k, _ in self.conv2.named_parameters():
            dont_train_those.append(k)

        # print(params)
        dont_train_those = []
        for k, _ in self.conv3.named_parameters():
            dont_train_those.append(k)
        # ---------------------------------------------------------------------------------------------
        params = []
        for k, v in self.named_parameters():
            if k not in dont_train_those:
                params.append(v)

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.05,
        )
        # num_gpus = 3
        # num_train_steps = len(self.train_dataset) // (self.hparams.batch_size * num_gpus) * self.trainer.max_epochs
        # print("\n num_train_steps: ", num_train_steps)
        # num_warmup_steps = int(self.hparams.warmup_ratio * num_train_steps)
        # print("\n num_warmup_steps: ", num_warmup_steps)
        #
        # lr_scheduler = {
        #     "scheduler": get_cosine_schedule_with_warmup(
        #         optimizer,
        #         num_warmup_steps=num_warmup_steps,
        #         num_training_steps=num_train_steps,
        #         num_cycles=0.5,
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }
        return [optimizer] #, [lr_scheduler]

