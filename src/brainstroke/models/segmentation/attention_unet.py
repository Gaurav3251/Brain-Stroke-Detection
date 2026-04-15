import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.b(x)


class scSE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 8)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1), nn.Sigmoid(),
        )
        self.sse = nn.Sequential(
            nn.Conv2d(ch, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class EncBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = ConvBlock(i, o)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.conv(x)
        return self.pool(s), s


class DecBlock(nn.Module):
    def __init__(self, i, sk, o):
        super().__init__()
        self.up = nn.ConvTranspose2d(i, o, 2, stride=2)
        self.conv = ConvBlock(o + sk, o)
        self.att = scSE(o)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv(torch.cat([x, skip], 1))
        return self.att(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.encoders = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.encoders.append(EncBlock(ch, f))
            ch = f
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        self.decoders = nn.ModuleList()
        dec_ch = features[-1] * 2
        for f in reversed(features):
            self.decoders.append(DecBlock(dec_ch, f, f))
            dec_ch = f
        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        sz = x.shape[2:]
        skips = []
        for enc in self.encoders:
            x, s = enc(x)
            skips.append(s)
        x = self.bottleneck(x)
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])
        out = self.final(x)
        if out.shape[2:] != sz:
            out = F.interpolate(out, sz, mode="bilinear", align_corners=False)
        return out
