import torch
import pytorch_lightning as pl


class SeamRemoval(pl.LightningModule):
    def __init__(
        self,
    ):
        super(SeamRemoval, self).__init__()
        self.seam_removal_conv3d_1 = torch.nn.Conv3d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            padding_mode="replicate",
        )
        self.seam_removal_conv3d_4 = torch.nn.Conv3d(
            in_channels=8,
            out_channels=1,
            kernel_size=(5, 5, 5),
            stride=(1, 1, 1),
            padding=(2, 2, 2),
            padding_mode="replicate",
        )

    def forward(self, decoded_trans_output_collected_u: torch.Tensor) -> torch.Tensor:

        x = decoded_trans_output_collected_u.detach()  # don't backprop across here
        x = self.seam_removal_conv3d_1(x)
        x = torch.relu(x)
        x = self.seam_removal_conv3d_4(x)
        x = x.squeeze(1)
        return x
