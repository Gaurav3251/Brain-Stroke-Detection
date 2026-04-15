import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.config import NUM_CLASSES

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False


class SwinDecBlock(nn.Module):
    def __init__(self, i, sk, o):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(i, o, 1, bias=False),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(o + sk, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class SwinUNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, img_size=224, pretrained=True):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm required for SwinUNet")
        self._is_swin_unet = True
        self.input_size = img_size
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            img_size=img_size,
        )
        enc_ch = self.encoder.feature_info.channels()
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(enc_ch[-1], 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
        )
        self.cls_head = nn.Linear(512, num_classes)
        dec_ch = [256, 128, 64, 32]
        self.decoders = nn.ModuleList()
        in_c = enc_ch[-1]
        for i, out_c in enumerate(dec_ch):
            sk = enc_ch[-(i + 2)] if i < len(enc_ch) - 1 else 0
            if sk > 0:
                self.decoders.append(SwinDecBlock(in_c, sk, out_c))
            else:
                self.decoders.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                ))
            in_c = out_c
        self.seg_head = nn.Conv2d(dec_ch[-1], 1, 1)
        self._enc_ch = enc_ch

    def _reshape(self, f):
        if f.dim() == 4 and f.shape[-1] != f.shape[-3]:
            return f.permute(0, 3, 1, 2).contiguous()
        return f

    def forward(self, x):
        sz = x.shape[2:]
        feats = [self._reshape(f) for f in self.encoder(x)]
        cls_v = self.bottleneck(feats[-1])
        cls_o = self.cls_head(cls_v)
        xd = feats[-1]
        for i, dec in enumerate(self.decoders):
            si = len(feats) - 2 - i
            if si >= 0 and isinstance(dec, SwinDecBlock):
                xd = dec(xd, feats[si])
            else:
                xd = dec(xd)
        seg_o = F.interpolate(self.seg_head(xd), sz, mode="bilinear", align_corners=False)
        return cls_o, seg_o
