import sys
from typing import Tuple, Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

sys.path.append("..")
import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from training.train_dataset import LMDBOBJAVERSEPARTIALVIEWS
from p_vae.pvae import SDFtoSDF
from utils import transformer_visualizations as tv
from utils import plot_march_fns as pmt_fns
from utils import sub_voxel_related_fns as pp_fns
from utils.positional_encoder_class import MYPositionalEncoder3D
from utils import encoder_decoder_loading as ed
from transformers.optimization import get_cosine_schedule_with_warmup
from utils import encoder_related_fns as enc_fns
from utils.m_cube_fns import make_mcubes_from_voxels_obj_with_pad


# ----------------------------------------------------------------------------------------------------------------------------------------------------
class CompletePartialScans(pl.LightningModule):
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
        num_warmup_steps: int,
        num_training_steps: int,
    ):
        super(CompletePartialScans, self).__init__()
        self.save_hyperparameters()

        self.num_warmup_steps = num_warmup_steps
        self.num_train_steps = num_training_steps

        self.number_of_sub_voxels = 64
        self.l1_loss = nn.L1Loss(reduction="mean")

        number_of_sub_voxels = self.hparams.resolution // self.hparams.target_resolution
        self.number_of_sub_voxels = (
            number_of_sub_voxels * number_of_sub_voxels * number_of_sub_voxels
        )

        # To visualize only these objects in tensorboard. If you want to visualize everything , remove them from if condition in plot_march_and_login_tensorboard function
        self.my_selected_indices = [
            44,
            20,
            75,
            45,
            30,
            10,
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
        ]

        if self.hparams.pre_trained:
            print("\n pre_trained: ", pre_trained)
            pre_trained_vae = SDFtoSDF.load_from_checkpoint(
                vae_checkpoint_path, map_location="cpu"
            )
            pre_trained_vae.freeze()
            pre_trained_vae.train(False)

            self.fdecoder = ed.load_decoder_from_checkpoint(pre_trained_vae, latent_dim)
            self.fencoder = ed.load_encoder_from_checkpoint(pre_trained_vae, latent_dim)

        self.penc_channels = 8 * self.hparams.latent_dim * 2
        self.positional_encoder_3d = MYPositionalEncoder3D(self.penc_channels)

        self.regular_transformer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                self.hparams.dim_size,
                self.hparams.heads,
                dim_feedforward=self.hparams.dim_size,
                batch_first=True,
            ),
            num_layers=self.hparams.layers,
            norm=torch.nn.LayerNorm(self.hparams.dim_size),
        )

        self.mapping_down = nn.Linear(
            self.penc_channels + 2 * 8 * self.hparams.latent_dim, self.hparams.dim_size
        )
        self.mapping_up = nn.Linear(self.hparams.dim_size, 8 * self.hparams.latent_dim)

        # [1,128,128,128]
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        # [128,32,32,32]
        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        # [64,16,16,16]
        self.conv3 = nn.Sequential()

    def call_transformer_and_mapping_layers(
        self, transformer_input_sequence: torch.Tensor
    ) -> torch.Tensor:
        transformer_input_sequence = transformer_input_sequence

        transformer_input_sequence_down = self.mapping_down(transformer_input_sequence)
        transformer_output_sequence = self.regular_transformer(
            transformer_input_sequence_down
        )
        transformer_output_sequence_up = self.mapping_up(transformer_output_sequence)

        return transformer_output_sequence_up

    def forward(
        self, sdf_latent_codes: torch.Tensor, uncertainty_latent_codes: torch.Tensor
    ) -> torch.Tensor:
        # This part is the same for training and validation
        batch_size = sdf_latent_codes.shape[0]
        concatenated_latent_codes = torch.cat(
            (sdf_latent_codes, uncertainty_latent_codes), dim=2
        ).to(device=sdf_latent_codes.device)
        concatenated_latent_codes_reshaped = concatenated_latent_codes.reshape(
            [batch_size, self.number_of_sub_voxels, 2 * 8 * self.hparams.latent_dim]
        )
        z_positionally_encoded_re = self.positional_encoder_3d(
            shape_of_positions=[batch_size, 4, 4, 4, self.penc_channels]
        )
        assert (
            z_positionally_encoded_re.shape == concatenated_latent_codes_reshaped.shape
        )
        transformer_input_sequence = torch.cat(
            (z_positionally_encoded_re, concatenated_latent_codes_reshaped), dim=2
        ).to(device=z_positionally_encoded_re.device)
        transformer_output_sequence = self.call_transformer_and_mapping_layers(
            transformer_input_sequence
        )

        return transformer_output_sequence

    def encode_stuff(
        self,
        gt_sdf_sub_voxels: torch.Tensor,
        gt_uncertainty_combined_voxel: torch.Tensor,
        train: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = gt_sdf_sub_voxels.shape[0]
        # embed the sdfs -> sdf_latent_code
        sdf_latent_codes, sdf_std, sdf_var = enc_fns.prep_32cube_sub_voxels_and_encode(
            self.fencoder,
            gt_sdf_sub_voxels,
            self.number_of_sub_voxels,
            self.hparams.latent_dim,
            self.hparams.target_resolution,
            train,
        )

        # embed the uncertainty values -> uncertainty_latent_code using a conv3D with stride to shrink its dimensionality
        # adding a batch dim first
        gt_uncertainty_combined_voxel_batched = gt_uncertainty_combined_voxel.unsqueeze(
            1
        )
        assert gt_uncertainty_combined_voxel_batched.shape == (
            batch_size,
            1,
            self.hparams.resolution,
            self.hparams.resolution,
            self.hparams.resolution,
        )

        uncertainty_latent_codes = self.conv1(gt_uncertainty_combined_voxel_batched)
        uncertainty_latent_codes = self.conv2(uncertainty_latent_codes)
        uncertainty_latent_codes = self.conv3(uncertainty_latent_codes)

        # [64,16,16,16]
        uncertainty_latent_codes = uncertainty_latent_codes.reshape(
            [-1, 64, 4, 4, 4, 4, 4, 4]
        )
        # 4x4x4 xyz patches to the front, channels to back
        # Batch, patch xyz, within-patch xyz, channels
        uncertainty_latent_codes = uncertainty_latent_codes.permute(
            [0, 2, 4, 6, 3, 5, 7, 1]
        )  # to put spatial dims upfront to align with sdf for concatenation

        uncertainty_latent_codes_r = uncertainty_latent_codes.reshape(
            [batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2]
        )
        return (sdf_latent_codes, uncertainty_latent_codes_r)

    def fwd(
        self, batch: list, train: bool, val: bool, test: bool
    ) -> tuple[dict[str, Any], tuple[Tensor, Tensor, Tensor]]:
        (
            object_indices,
            mesh_file_name,
            mesh_name,
            folder_name,
            combined_sdf_voxel,
            combined_uncertainty_voxel,
            gt_sdf_latent_codes,
        ) = batch
        batch_size = object_indices.shape[0]
        # -----------------------------------------------------------------------------------------------------------------------------------------
        combined_sdf_sub_voxels = pp_fns.sub_divide_gt_and_normalize(
            combined_sdf_voxel,
            self.number_of_sub_voxels,
            self.hparams.target_resolution,
        )
        # uncertainty values are normalized to [-1,1] already in dataset class
        sdf_latent_codes, uncertainty_latent_codes = self.encode_stuff(
            combined_sdf_sub_voxels, combined_uncertainty_voxel, False
        )
        transformer_output_sequence_up = self.forward(
            sdf_latent_codes, uncertainty_latent_codes
        )
        transformer_output_sequence_up_reshaped = (
            transformer_output_sequence_up.reshape(
                batch_size, self.number_of_sub_voxels, -1
            )
        )

        # For loss calculation against the gt_sdf_latent_code that is precalculated from LMDB
        gt_sdf_latent_codes_reshaped = gt_sdf_latent_codes.reshape(
            [batch_size, self.number_of_sub_voxels, 8 * self.hparams.latent_dim]
        )
        assert (
            transformer_output_sequence_up_reshaped.shape
            == gt_sdf_latent_codes_reshaped.shape
        )

        # calculate losses:
        loss_l1 = self.l1_loss(
            transformer_output_sequence_up_reshaped, gt_sdf_latent_codes_reshaped
        )
        # log losses
        if train:
            loss_l1 = self.l1_loss(
                transformer_output_sequence_up_reshaped, gt_sdf_latent_codes_reshaped
            )
            self.log(
                "train_loss",
                loss_l1,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            loss_dict = {"loss": loss_l1}

        elif val:
            self.log(
                "val_loss",
                loss_l1,
                batch_size=self.hparams.val_batch_size,
                sync_dist=True,
            )
            loss_dict = {"loss": loss_l1}

        elif test:
            loss_dict = {"loss": loss_l1}

        else:
            raise ("\ninvalid stage!")

        return (
            loss_dict,
            (
                sdf_latent_codes,
                uncertainty_latent_codes,
                transformer_output_sequence_up,
            ),
        )

    def training_step(self, batch: list, batch_idx: int) -> dict:
        # (
        #     object_indices,
        #     mesh_file_name, mesh_name, folder_name, combined_sdf_voxel, combined_uncertainty_voxel, gt_sdf_latent_codes
        # ) = batch
        loss_dict, stuff = self.fwd(batch, train=True, val=False, test=False)

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
            mesh_file_name,
            mesh_name,
            folder_name,
            combined_sdf_voxel,
            combined_uncertainty_voxel,
            gt_sdf_latent_codes,
        ) = batch

        batch_size = object_indices.shape[0]
        assert object_indices.shape == (self.hparams.val_batch_size,)
        loss_dict, stuff = self.fwd(batch, train=False, val=True, test=False)

        sdf_latent_codes, uncertainty_latent_codes, transformer_output_sequence_up = (
            stuff
        )

        # just for vis
        transformer_output_sequence_up_reshaped = (
            transformer_output_sequence_up.reshape(
                batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2
            )
        )
        # visualization and tensorboard---------------------------
        dict_arguments_for_vis = {
            "InputSdfLatentCodes": sdf_latent_codes,
            "TransformerOutput": transformer_output_sequence_up_reshaped,
            "GT_sdfLatentCodes": gt_sdf_latent_codes,
        }

        dict_arguments_of_variables = {
            "number_of_sub_voxels": self.number_of_sub_voxels,
            "latent_dim": self.hparams.latent_dim,
            "target_resolution": self.hparams.target_resolution,
            "resolution": self.hparams.resolution,
            "batch_size": batch_size,
            "Uncertainty": combined_uncertainty_voxel,
        }

        self.plot_march_and_login_tensorboard(
            dict_arguments_for_vis,
            dict_arguments_of_variables,
            object_indices,
            batch_size,
        )

    def test_step(self, batch: list) -> None:
        (
            object_indices,
            mesh_file_name,
            mesh_name,
            folder_name,
            combined_sdf_voxel,
            combined_uncertainty_voxel,
            gt_sdf_latent_codes,
        ) = batch
        batch_size = object_indices.shape[0]
        assert object_indices.shape == (self.hparams.val_batch_size,)

        loss_dict, stuff = self.fwd(batch, train=False, val=False, test=True)
        sdf_latent_codes, uncertainty_latent_codes, transformer_output_sequence_up = (
            stuff
        )
        # I want them to be marched, so I get to the results.

        # just for vis
        transformer_output_sequence_up_reshaped = (
            transformer_output_sequence_up.reshape(
                batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2
            )
        )
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
        dict_data_vis = pmt_fns.generate_any_data_for_plotting(
            dict_args_vis, dict_args_variables, self.fdecoder
        )
        dict_args_eval: dict = {}
        for b in range(batch_size):
            selected_index = object_indices.detach().item()
            collected_data_dict_for_plotting = (
                pmt_fns.collect_any_generated_data_for_plotting(
                    dict_data_vis, batch_idx=b
                )
            )
            keys = [key for key in collected_data_dict_for_plotting.keys()]
            for i in range(len(keys)):
                current_key = keys[i]
                collected_data_current = collected_data_dict_for_plotting.get(
                    current_key
                )
                export_file_name_obj = (
                    self.hparams.obj_dir
                    + str(current_key)
                    + "-"
                    + "ObjID="
                    + str(selected_index)
                    + "_"
                    + ".obj"
                )
                make_mcubes_from_voxels_obj_with_pad(
                    collected_data_current, export_file_name_obj
                )
                print()
                if export_file_name_obj.startswith(
                    "Input"
                ) and export_file_name_obj.endswith(".obj"):
                    dict_args_eval["Input"] = export_file_name_obj
                if export_file_name_obj.startswith(
                    "Transformer"
                ) and export_file_name_obj.endswith(".obj"):
                    dict_args_eval["Transformer"] = export_file_name_obj
                if export_file_name_obj.startswith(
                    "GT"
                ) and export_file_name_obj.endswith(".obj"):
                    dict_args_eval["GT"] = export_file_name_obj
                # now evaluate this object:
                dict_args_eval["Completed_voxel"] = (
                    transformer_output_sequence_up_reshaped[b]
                )
                dict_args_eval["GT_voxel"] = gt_sdf_latent_codes[b]
                dict_args_eval["Input_voxel"] = sdf_latent_codes[b]
                dict_args_eval["Object_index"] = object_indices[b]
                dict_args_eval["Object_index"] = self.hparams.num_samples

    def plot_march_and_login_tensorboard(
        self,
        dict_arguments_for_vis: dict,
        dict_arguments_of_variables: dict,
        object_indices: torch.Tensor,
        batch_size: int,
    ) -> None:
        # this is decoded now
        data_dict_for_vis = pmt_fns.generate_any_data_for_plotting(
            dict_arguments_for_vis, dict_arguments_of_variables, self.fdecoder
        )
        # Diff------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        diff_transformer_output_vs_latent_codes = torch.subtract(
            data_dict_for_vis["TransformerOutput"],
            data_dict_for_vis["GT_sdfLatentCodes"],
        )
        data_dict_for_vis["diff_transformerOutput_sdfLatentCodes"] = (
            diff_transformer_output_vs_latent_codes
        )

        for b in range(batch_size):
            selected_index = object_indices[b].detach().cpu().item()
            if (
                selected_index in self.my_selected_indices
            ):  # if you want to visualize all the objects in tensorboard, remove the if-condition
                collected_data_dict_for_plotting = (
                    pmt_fns.collect_any_generated_data_for_plotting(
                        data_dict_for_vis, batch_idx=b
                    )
                )
                collected_data_dict_for_plotting["Uncertainty"] = (
                    dict_arguments_of_variables["Uncertainty"][b]
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                plots = tv.generate_plot_for_given_dict_of_items(
                    collected_data_dict_for_plotting,
                    self.hparams.resolution,
                    number_of_slices=2,
                    plot_scale_factor=2,
                    plot_range=2,
                )
                self.login_to_tensorboard(plots, selected_index, number_of_slices=2)
                pmt_fns.march_any_results_every_n_epoch(
                    collected_data_dict_for_plotting,
                    selected_index,
                    self.current_epoch,
                    self.trainer.global_step,
                    self.hparams.marching_cube_result_dir,
                )

    def login_to_tensorboard(
        self, plots: list, selected_index: int, number_of_slices: int
    ):
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

        self.train_dataset = LMDBOBJAVERSEPARTIALVIEWS(
            self.hparams.mesh_path,
            self.hparams.train_lmdb_path,
            self.hparams.marching_cube_result_dir,
            self.hparams.image_resolution,
            self.hparams.resolution,
            device=self.device,
        )

        print("\n train_dataset len:", len(self.train_dataset))

        # for validation while training
        self.val_dataset = LMDBOBJAVERSEPARTIALVIEWS(
            self.hparams.mesh_path,
            self.hparams.val_lmdb_path,
            self.hparams.marching_cube_result_dir,
            self.hparams.image_resolution,
            self.hparams.resolution,
            device=self.device,
        )
        # self.val_dataset.len = 200
        print("\n val_dataset len:", len(self.val_dataset))

        # for evaluation
        self.test_dataset = LMDBOBJAVERSEPARTIALVIEWS(
            self.hparams.mesh_path,
            self.hparams.test_lmdb_path,
            self.hparams.marching_cube_result_dir,
            self.hparams.image_resolution,
            self.hparams.resolution,
            device=self.device,
        )

        print("\n test_dataset len:", len(self.test_dataset))

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
        # we exclude fdecoer params from optimizer explicitly!
        dont_train_those = []
        for k, _ in self.fdecoder.named_parameters():
            dont_train_those.append(k)
        params = []

        for k, v in self.named_parameters():
            if k not in dont_train_those:
                params.append(v)
        # print(params)

        # we exclude fencoder params from optimizer explicitly!
        dont_train_those = []
        for k, _ in self.fencoder.named_parameters():
            dont_train_those.append(k)
        params = []

        for k, v in self.named_parameters():
            if k not in dont_train_those:
                params.append(v)
        # print(params)

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.05,
        )

        lr_scheduler = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_train_steps,
                num_cycles=0.5,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
