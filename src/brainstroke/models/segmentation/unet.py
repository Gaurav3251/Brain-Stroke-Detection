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

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=None, deep_sup=True):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.deep_sup = deep_sup
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
        if deep_sup:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(f, out_ch, 1) for f in reversed(features)
            ])

    def forward(self, x):
        sz = x.shape[2:]
        skips = []
        for enc in self.encoders:
            x, s = enc(x)
            skips.append(s)
        x = self.bottleneck(x)
        ds_outs = []
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])
            if self.deep_sup:
                ds_outs.append(F.interpolate(self.ds_heads[i](x), sz,
                                             mode="bilinear", align_corners=False))
        out = self.final(x)
        if self.training and self.deep_sup:
            return out, ds_outs
        return out
