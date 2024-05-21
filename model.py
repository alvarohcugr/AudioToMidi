import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from constants import N_BINS_PER_OCTAVE

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Definir la arquitectura de la red neuronal
class HarmonicStacking(nn.Module):
    def __init__(self, harmonics):
        super(HarmonicStacking, self).__init__()
        self.harmonics=harmonics
        self.shifts=[int(np.round(N_BINS_PER_OCTAVE*np.log2(h))) for h in harmonics]

    def forward(self, x):
        #x = x.to(self.fc1.weight.dtype)  # Convertir x al mismo tipo de datos que self.fc1.weight
        n_batch, n_bins, n_frames= x.shape
        shifted=torch.zeros((n_batch, len(self.harmonics), n_bins, n_frames)).to(device)
        for i, s in enumerate(self.shifts):
          if s==0:
            shifted[:, i]=x
          elif s>0:
            shifted[:, i, :-s, :]=x[:, s:, :]
          elif s<0:
            shifted[:, i, -s:, :]=x[:, :s, :]
        return shifted
class UNet(nn.Module):
    def __init__(self, n_classes, harmonics):
        super(UNet, self).__init__()
        self.hs=HarmonicStacking(harmonics)
        self.inc = DoubleConv(len(harmonics), 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes, stride=(3,1))

        self.outc_o = OutConv(64, n_classes, stride=(3,1))

    def forward(self, x):
        xh=self.hs(x)
        x1 = self.inc(xh)
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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        #x = x.to(self.max_pool_conv.weight.dtype)
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.conv(x)
