""" Full assembly of the parts to form the complete network """

from .unet_parts_wavelet import *
from .resizer import *
from .SAA import *
from omegaconf import DictConfig
import hydra


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src


        
# class out_conv_last(nn.Module):
#     def __init__(self, ch1, ch2, ch3, n_classes):
#         super().__init__()
#         self.out_conv = nn.Conv2d(ch1 + ch2 + ch3, n_classes, 1)
        
#     def forward(self, x1, x2, x3):
#             x2 = F.upsample(x2, size=x1.shape[2:], mode='bilinear')
#             x3 = F.upsample(x3, size=x1.shape[2:], mode='bilinear')
        
#             out = torch.cat((x1, x2, x3), dim=1)
#             out = self.out_conv(out)
#             return out
    
        
    
    

class UNet_F_wave(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(UNet_F_wave, self).__init__()
        self.n_classes = cfg.model.n_classes
        self.in_resizer = Resizer(cfg)
       # self.saa = SAA(cfg)
        self.layer = cfg.model.layer
        self.mode = cfg.model.mode
        self.n_channels = cfg.model.n_channels

        if self.mode == 1:
            if self.layer == 4:
                self.inc = DoubleConv(self.n_channels, 64)
                # self.inc = DoubleConv(1, 64)
                self.down1 = (part_dwt1(64, 128))
                self.down2 = (part_dwt1(128, 256))
                self.down3 = (part_dwt1(256, 512))
                self.down4 = (last_dwt1(512, 1024, cfg.model.se))

                self.up4 = (part_iwt1_fuse(128, 512, cfg.model.bilinear))
                # self.conv4 = nn.Conv2d(512, self.n_classes, 1)
                # self.conv4 = (out_conv_last(512, 256, 128, self.n_classes))
                self.conv4 = (out_conv(512, 256, self.n_classes))

                self.up3 = (part_iwt1(512, 256, cfg.model.bilinear))
                # self.conv3 = nn.Conv2d(256, self.n_classes, 1)
                self.conv3 = (out_conv(256, 128, self.n_classes))
                
                self.up2 = (part_iwt1(256, 128, cfg.model.bilinear))
                # self.conv2 = nn.Conv2d(128, self.n_classes, 1)
                self.conv2 = (out_conv(128, 64, self.n_classes))

                self.up1 = (part_iwt1(128, 64, cfg.model.bilinear))
                self.conv1 = nn.Conv2d(64, self.n_classes, 1)

                self.out = (OutConv(self.n_classes * self.layer, self.n_classes, cfg.data.size, cfg.data.interpolate_mode))

        

    def forward(self, x):
        x_resized = self.in_resizer(x)
        # x_interp = self.interpolate(x)
        # x = self.inc(torch.cat([x_resizer, x_interp], dim=1))
        # x = self.inc(torch.add(x_resizer, x_interp))
       # x_interp = self.interpolate(x)

        # x = self.saa(x)
        x = self.inc(x_resized)


        if self.mode == 1:
            if self.layer == 4:
                x1, x1_dwt = self.down1(x)
                x2, x2_dwt = self.down2(x1)
                x3, x3_dwt = self.down3(x2)
                x4_dwt = self.down4(x3)

                x4_up = self.up4(x1, x4_dwt)
                x3_up = self.up3(x4_up, x3_dwt)
                x2_up = self.up2(x3_up, x2_dwt)

                x1_up = self.up1(x2_up, x1_dwt)
                x1_out = self.conv1(x1_up)

                x2_out = self.conv2(x2_up, x1_up)
                x2_out = _upsample_like(x2_out, x1_out)

                x3_out = self.conv3(x3_up, x2_up)
                x3_out = _upsample_like(x3_out, x1_out)
                                
                # x4_out = self.conv4(x4_up, x3_up, x1)
                x4_out = self.conv4(x4_up, x3_up)
                x4_out = _upsample_like(x4_out, x1_out)

                out = torch.cat((x1_out,x2_out,x3_out,x4_out),1)

                out = self.out(out)

 

        # return out, x_resizer
        return out, x_resized
