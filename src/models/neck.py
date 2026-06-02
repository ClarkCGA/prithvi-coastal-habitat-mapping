import torch
import torch.nn as nn
import torch.nn.functional as F

class Norm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(Norm2d, self).__init__()
        # LayerNorm that operates on each channel, so we normalize across channels only
        self.ln = nn.LayerNorm((num_features,), eps=eps, elementwise_affine=True)

    def forward(self, x):
        # x has shape [batch_size, channels, height, width]
        # First permute to [batch_size, height, width, channels] for LayerNorm
        x = x.permute(0, 2, 3, 1).contiguous()

        # Apply LayerNorm
        x = self.ln(x)

        # Permute back to original shape [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

# Option 1: Use Transposed convolution with reflective padding to go from 16x16 to 28x28
class FPN1(nn.Module):
    def __init__(self,in_channel,out_channel, patch_size):
        super(FPN1, self).__init__()
        self.patch_size = patch_size

        if self.patch_size[1] == 14:
            # Grid 16x16 --> 28x28 --> 56x56
            self.fpn1 = nn.Sequential(
                nn.ReflectionPad2d(2),
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2, 2), stride=(2, 2)),
                #nn.ReflectionPad2d(1),
                #nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(3, 3), stride=(2, 2))),
                Norm2d(out_channel),
                nn.GELU(),
                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(2, 2), stride=(2, 2))
            )
        elif self.patch_size[1] == 16:
            # Grid 14x14 → 28x28 --> 56x56
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2, 2), stride=(2, 2)),
                Norm2d(out_channel),
                nn.GELU(),
                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(2, 2), stride=(2, 2))
            )
    def forward(self, x):
        return self.fpn1(x)

# Option 2: Using bilinear interpolation
class FPN1(nn.Module):
    def __init__(self, in_channel, out_channel, patch_size):
        super(FPN1, self).__init__()
        
        self.patch_size = patch_size
        self.target = (28, 28)

        self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2, 2), stride=(2, 2))
        self.deconv2 = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(2, 2), stride=(2, 2))
        self.norm = Norm2d(out_channel)
        self.act = nn.GELU()

    def forward(self, x):
        # Grid 16x16 --> 28x28 --> 56x56
        if self.patch_size[1] == 14:
            x = self.proj(x)
            x = F.interpolate(x, size=self.target, mode='bilinear', align_corners=False)
            x = self.act(self.norm(self.conv(x)))
            x = self.act(self.norm(self.deconv2(x)))
            
        if self.patch_size[1] == 16:
            # Grid 14x14 → 28x28 --> 56x56
            x = self.act(self.norm(self.deconv1(x)))
            x = self.act(self.norm(self.deconv2(x)))

        return x

# Option 3: Making option 2 less deterministic
class FPN1(nn.Module):
    def __init__(self, in_channel, out_channel, patch_size):
        super(FPN1, self).__init__()
        
        upscale_factor = 2
        self.patch_size = patch_size
        self.target = (28, 28)

        self.conv = nn.Conv2d(in_channel, (upscale_factor ** 2) * out_channel, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        self.deconv1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2, 2), stride=(2, 2))
        self.deconv2 = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(2, 2), stride=(2, 2))
        
        self.norm = Norm2d(out_channel)
        self.act = nn.GELU()


    def forward(self, x):
        # Grid 16x16 --> 28x28 --> 56x56
        if self.patch_size[1] == 14:
            x = self.act(self.norm(self.conv(x)))
            x = self.pixel_shuffle(x)
            x = F.interpolate(x, size=self.target, mode='bilinear', align_corners=False)
            x = self.norm(self.conv1x1(x))

            
        if self.patch_size[1] == 16:
            # Grid 14x14 → 28x28 --> 56x56
            x = self.act(self.norm(self.deconv1(x)))
            x = self.act(self.norm(self.deconv2(x)))

        return x


class FPN2(nn.Module):
    def __init__(self,out_channel):
        super(FPN2, self).__init__()
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(2, 2), stride=(2, 2)),
            Norm2d(out_channel,eps=1e-6),  # Custom Norm2d with LayerNorm
            nn.GELU(),
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, x):
        return self.fpn1(x)

    

class Neck(nn.Module):
    def __init__(self,embed_size, patch_size):
        super(Neck, self).__init__()
        
        self.embed_dim=embed_size

        # Conv layer to upscale the token grid to the desired segmented image size
        self.fpn1=FPN1(self.embed_dim,self.embed_dim, patch_size)
        self.fpn2=FPN2(self.embed_dim)
       

    def forward(self, x):
        
        x = self.fpn1(x)
        x = self.fpn2(x)
        
        #print("x shape",x.shape)
        
        return x