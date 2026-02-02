import torch
import pytorch_lightning as pl
from positional_encodings.torch_encodings import (
    PositionalEncoding3D, PositionalEncoding1D
)
from einops import rearrange

class MYPositionalEncoder3D(pl.LightningModule):
    def __init__(self, channels):
        super(MYPositionalEncoder3D, self).__init__()
        self.positional_encoder = PositionalEncoding3D(channels)

    def forward(self, shape_of_positions: list) -> torch.Tensor:
        # B, SqL_l, Ch = masked_optimized_latent_codes_reshaped.shape
        batch_size = shape_of_positions[0]
        D = shape_of_positions[1]
        H = shape_of_positions[2]
        W = shape_of_positions[3]
        Ch = shape_of_positions[4]

        z = torch.zeros((batch_size, D, H, W, Ch), dtype=torch.float32, device=self.device)
        z_positionally_encoded = self.positional_encoder(z)
        del z
        z_positionally_encoded_re = rearrange(z_positionally_encoded, "B D H W Ch -> B (D H W) Ch").to(self.device)
        del z_positionally_encoded

        return z_positionally_encoded_re

class MYPositionalEncoder4D(pl.LightningModule):
    def __init__(self, channels):
        super(MYPositionalEncoder4D, self).__init__()
        assert channels % 4 == 0
        self.positional_encoder3D = PositionalEncoding3D(channels*3//4)
        self.positional_encoder1D = PositionalEncoding1D(channels*1//4)

    def forward(self, shape_of_3dpositions: list, shape_of_1dpositions: list) -> torch.Tensor:
        # B, SqL_l, Ch = masked_optimized_latent_codes_reshaped.shape
        batch_size = shape_of_3dpositions[0]
        D = shape_of_3dpositions[1]
        H = shape_of_3dpositions[2]
        W = shape_of_3dpositions[3]
        Ch = shape_of_3dpositions[4]

        # 3D--------------------------------------------------------------------------------------------------
        z3d = torch.zeros((batch_size, D, H, W, Ch), dtype=torch.float32, device=self.device)
        z3d_positionally_encoded = self.positional_encoder3D(z3d)
        del z3d
        z3d_positionally_encoded_re = rearrange(z3d_positionally_encoded, "B D H W Ch -> B (D H W) Ch").to(self.device)
        del z3d_positionally_encoded
        # 1D---------------------------------------------------------------------------------------------------
        batch_size_ = shape_of_1dpositions[0]
        DHW_ = shape_of_1dpositions[1]
        Ch_ = shape_of_1dpositions[2]

        z1d = torch.zeros((batch_size_, DHW_, Ch_), dtype=torch.float32, device=self.device)
        z1d_positionally_encoded = self.positional_encoder3D(z1d)
        del z1d
        z1d_positionally_encoded_re = rearrange(z1d_positionally_encoded, "B DHW Ch -> B (DHW) Ch").to(self.device)
        del z1d_positionally_encoded

        assert (z3d_positionally_encoded_re.shape == z1d_positionally_encoded_re.shape)
        # concat
        z4d_positionally_encoded = torch.cat((z3d_positionally_encoded_re, z1d_positionally_encoded_re), dim=1)

        return z4d_positionally_encoded

