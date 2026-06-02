import torch
import torch.nn as nn
import torch.nn.functional as F
from .Prithvi import MaskedAutoencoderViT

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(UNetEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv3d(hidden_channels * 4, hidden_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 8, hidden_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        return enc1, enc2, enc3, enc4


class UNetDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(UNetDecoder, self).__init__()

        self.decoder4 = nn.Sequential(
            nn.Conv3d(hidden_channels * 16, hidden_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 8, hidden_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 8),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv3d(hidden_channels * 12, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv3d(hidden_channels * 6, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv3d(hidden_channels * 3, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        self.final_conv = nn.Conv3d(hidden_channels, out_channels, kernel_size=(4, 1, 1))

        self.ReLU = nn.ReLU()

    def forward(self, enc1, enc2, enc3, enc4, bottleneck):
        
        dec4 = torch.cat((bottleneck, enc4), dim=1)

        dec4 = self.decoder4(dec4)


        dec3 = torch.cat((dec4, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = torch.cat((dec3, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = torch.cat((dec2, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        final = self.final_conv(dec1)

        output = self.ReLU(final)
        return output

class UNetWithTransformer(nn.Module):
    """
    U-Net adapter with the following flow:
    x -> unet_enc -> adapter -> unet_dec
    x -> embed -> backbone
    """

    def __init__(self, n_channels, n_classes,n_frame,embed_size,input_size,
                 patch_size,prithvi_weight,hidden_channels: int = 16):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pr_weight=prithvi_weight
        
        self.n_frame=n_frame
        self.input_size=input_size
        self.embed_size=embed_size
        self.patch_size=patch_size

        self.tubelet_size=1
        self.depth = 24
        self.num_heads = 16
        self.decoder_embed_dim=512
        self.decoder_depth=8
        self.decoder_num_heads=16
        self.mlp_ratio = 4.0
        self.norm_layer= nn.LayerNorm
        self.norm_pix_loss = False

        self.hidden_channels = hidden_channels

        self.unet_encoder = UNetEncoder(in_channels=n_channels, hidden_channels=hidden_channels)
        self.backbone: torch.nn.Module = MaskedAutoencoderViT(
            self.input_size, 
            self.patch_size,
            self.n_frame,
            self.tubelet_size,
            self.hidden_channels * 8,
            self.embed_size,
            self.depth,
            self.num_heads,
            self.decoder_embed_dim,
            self.decoder_depth,
            self.decoder_num_heads,
            self.mlp_ratio,
            self.norm_layer,
            self.norm_pix_loss, 
            self.pr_weight)
        
        self.unet_decoder = UNetDecoder(hidden_channels=hidden_channels, out_channels=1)

        # Freeze backbone weights
        self.backbone.requires_grad_(requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # assert input.dim() == 4  # BCHW
        # assert input.shape == torch.Size([2, 3, 224, 224])

        enc1, enc2, enc3, enc4 = self.unet_encoder(input)
        #imgs: torch.Tensor = enc4.unsqueeze(dim=2)  # Add time dimension, so BCHW -> BCTHW
        # assert imgs.shape == torch.Size([2, 64, 1, 224, 224])

        pred = self.backbone(x=enc4)
        pred_image: torch.Tensor = self.backbone.unpatchify(x=pred)
        # assert pred_image.shape == torch.Size([2, 64, 1, 224, 224])
        output: torch.Tensor = self.unet_decoder(
            enc1=enc1, enc2=enc2, enc3=enc3, enc4=enc4, bottleneck=pred_image
        )
        # assert output.shape == torch.Size([2, 3, 224, 224])
        output = torch.squeeze(output, dim=(2))
        
        return output