import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.b(x)


class UpBlock(nn.Module):
    def __init__(self, i, sk, o):
        super().__init__()
        self.up = nn.ConvTranspose2d(i, o, 2, stride=2)
        self.conv = ConvBlock(o + sk, o)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, pretrained=True):
        super().__init__()
        enc = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT if pretrained else None)
        self.enc1 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.enc2 = enc.layer1
        self.enc3 = enc.layer2
        self.enc4 = enc.layer3
        self.enc5 = enc.layer4

        self.bottleneck = ConvBlock(512, 512)
        self.dec4 = UpBlock(512, 256, 256)
        self.dec3 = UpBlock(256, 128, 128)
        self.dec2 = UpBlock(128, 64, 64)
        self.dec1 = UpBlock(64, 64, 32)
        self.final = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        sz = x.shape[2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bottleneck(e5)
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        out = self.final(d1)
        if out.shape[2:] != sz:
            out = F.interpolate(out, size=sz, mode="bilinear", align_corners=False)
        return out
