""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        num = 64
        self.inc = DoubleConv(n_channels, num)
        self.down1 = Down(num, num*2)
        self.down2 = Down(num*2 , num*4)
        self.down3 = Down(num*4 , num*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(num*8 , num*16 // factor)
        self.up1 = Up(num*16 , num*8 // factor, bilinear)
        self.up2 = Up(num*8 , num*4 // factor, bilinear)
        self.up3 = Up(num*4 , num*2 // factor, bilinear)
        self.up4 = Up(num*2 , num, bilinear)
        self.outc = OutConv(num, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
