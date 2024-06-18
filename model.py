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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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

class HarmonicStacking(nn.Module):
    def __init__(self, harmonics):
        super(HarmonicStacking, self).__init__()
        self.harmonics=harmonics
        self.shifts=[int(np.round(N_BINS_PER_OCTAVE*np.log2(h))) for h in harmonics]

    def forward(self, x):
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
# Definir la arquitectura de la red neuronal
class CNN_Note(nn.Module):
    def __init__(self, in_channels):
        super(CNN_Note, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=(2, 2), stride=(3, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 39), padding=(1, 19))
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=(1, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.to(self.conv1.weight.dtype)  # Convertir x al mismo tipo de datos que self.fc1.weight
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        return x

# Definir la arquitectura de la red neuronal
class CNN_Onset(nn.Module):
    def __init__(self, in_channels):
        super(CNN_Onset, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=(3, 1), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=65, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=(1, 1))
        # Inicialización de pesos Xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x, n):
        x = x.to(self.conv1.weight.dtype)  # Convertir x al mismo tipo de datos que self.fc1.weight
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = torch.cat((x, n), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x
class CNN_Model(nn.Module):
  def __init__(self, harmonics):
    super(CNN_Model, self).__init__()
    self.h_stack=HarmonicStacking(harmonics)
    self.onset=CNN_Onset(len(harmonics))
    self.note=CNN_Note(len(harmonics))
  def forward(self, x):
    x = self.h_stack(x)
    n = self.note(x)
    x = self.onset(x, torch.sigmoid(n))
    return n, x
