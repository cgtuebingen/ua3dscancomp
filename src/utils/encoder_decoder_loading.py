import sys
sys.path.append("..")

import torch
from Networks import frozen_encoder_decoder as fe

def load_encoder_from_checkpoint(pre_trained_model, latent_dim):
    params_encoder = pre_trained_model.encoder
    params_encoder.freeze()
    params_encoder.train(False)
    fencoder = fe.FrozenVSEncoderv9(params_encoder.to(pre_trained_model.device), latent_dim).to(pre_trained_model.device).train(False)
    fencoder.train(False)
    fencoder.freeze()
    # it is sequential, and the element 1 is always the batch_norm
    fencoder.convblock1.conv[1].track_running_stats = False
    fencoder.convblock1_1.conv[1].track_running_stats = False
    fencoder.convblock2.conv[1].track_running_stats = False
    fencoder.convblock2_1.conv[1].track_running_stats = False
    fencoder.convblock3.conv[1].track_running_stats = False
    fencoder.convblock3_1.conv[1].track_running_stats = False
    fencoder.convblock4.conv[1].track_running_stats = False
    fencoder.convblock4_1.conv[1].track_running_stats = False
    fencoder.convblock5.conv[1].track_running_stats = False
    fencoder.convblock5_1.conv[1].track_running_stats = False
    fencoder.convblock6.conv[1].track_running_stats = False
    fencoder.convblock6_1.conv[1].track_running_stats = False
    fencoder.convblock7.conv[1].track_running_stats = False
    fencoder.convblock7_1.conv[1].track_running_stats = False
    fencoder.convblock8.conv[1].track_running_stats = False
    fencoder.convblock8_1.conv[1].track_running_stats = False
    fencoder.convblock9.conv[1].track_running_stats = False
    fencoder.convblock9_1.conv[1].track_running_stats = False
    fencoder.convblock10.conv[1].track_running_stats = False
    fencoder.convblock10_1.conv[1].track_running_stats = False
    print(" encoder is restored from pre-trained VEncoder and is frozen")
    return fencoder


def load_decoder_from_checkpoint(pre_trained_model, latent_dim):
    params_decoder = pre_trained_model.decoder
    params_decoder.freeze()
    params_decoder.train(False)
    fdecoder = fe.FrozenVSDecoderv9(params_decoder.to(pre_trained_model.device), latent_dim).to(pre_trained_model.device).train(False)
    fdecoder.train(False)
    fdecoder.freeze()
    fdecoder.batchNorm3d5.track_running_stats = False
    fdecoder.batchNorm3d6.track_running_stats = False
    fdecoder.batchnorm3d7.track_running_stats = False
    # print( " decoder is restored from pre-trained VEncoder and is frozen")
    return fdecoder

def get_encoded_latent_code(self, sub_voxels: torch.Tensor) -> torch.Tensor:
    batch_size = sub_voxels.shape[0]
    #  Prep for Encoding--------------------------------------------------------------------------------------------------------------------
    # TODO : I was the reason for fucking you. dimension broadcasting, nasty python
    # Prepare sub_voxels for encoding and encode them to extract non-optimized_latent_codes-----------------------------------------------
    non_optimized_latent_codes = self.prep_sub_voxels_and_encode(sub_voxels)
    # Prepare for forward----------------------------------------------------------------------------------------------------------------
    non_optimized_latent_codes_reshaped = non_optimized_latent_codes.reshape(batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2)
    assert non_optimized_latent_codes_reshaped.shape == (batch_size, self.number_of_sub_voxels, self.hparams.latent_dim, 2, 2, 2)
    # Non-Optimized latent code collect------------------------------------------------------------------------------------------------------------
    # decoder--------------------------
    with torch.no_grad():
        decoded_non_optimized_latent_codes = self.fdecoder(non_optimized_latent_codes_reshaped).to(self.device)  # [128, 512, 2, 2, 2] -> [128, 1, 32, 32, 32]
    decoded_non_optimized_latent_codes_reshaped = decoded_non_optimized_latent_codes.reshape(
        [batch_size, self.number_of_sub_voxels, self.hparams.target_resolution, self.hparams.target_resolution, self.hparams.target_resolution]
    )
    # del decoded_latent_codes_reshaped
    collected_sub_voxels_decoded_non_optimized = collect_sub_voxels_to_voxel_with_batch(decoded_non_optimized_latent_codes_reshaped, self.hparams.resolution)
    assert collected_sub_voxels_decoded_non_optimized.shape == (batch_size, self.hparams.resolution, self.hparams.resolution, self.hparams.resolution)
    return non_optimized_latent_codes_reshaped
