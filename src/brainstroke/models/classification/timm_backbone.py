import torch
import torch.nn as nn
import warnings

from .base import BaseClassifier

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False


class TimmBackboneClassifier(BaseClassifier):
    def __init__(self, model_name: str, num_classes=3, dropout=0.3, pretrained=True, input_size=224):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm required for TimmBackboneClassifier")
        self.model_name = model_name
        self.input_size = input_size
        create_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "global_pool": "",
        }

        def _create_backbone(load_pretrained: bool):
            local_kwargs = dict(create_kwargs)
            local_kwargs["pretrained"] = load_pretrained
            try:
                return timm.create_model(
                    model_name,
                    **local_kwargs,
                    img_size=input_size,
                )
            except TypeError as exc:
                # Older or CNN-style timm models often reject img_size; the repo still
                # tracks the intended resize separately via self.input_size.
                if "img_size" not in str(exc):
                    raise
                return timm.create_model(model_name, **local_kwargs)

        try:
            self.backbone = _create_backbone(pretrained)
        except Exception as exc:
            if not pretrained:
                raise
            warnings.warn(
                f"Could not load pretrained weights for '{model_name}' ({exc}). "
                "Falling back to randomly initialized weights.",
                RuntimeWarning,
            )
            self.backbone = _create_backbone(False)
        self.feature_channels = getattr(self.backbone, "num_features", None)
        if self.feature_channels is None:
            self.feature_channels = self.backbone.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.feature_channels, num_classes)

    def _ensure_4d(self, feat):
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.dim() == 2:
            feat = feat[:, :, None, None]
        return feat

    def forward_features(self, x):
        if hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
        else:
            feat = self.backbone(x)
        return self._ensure_4d(feat)

    def forward_head(self, feat):
        return self.head(self.drop(self.pool(feat).flatten(1)))

    def cam_layer(self):
        """Return the last spatial feature layer for GradCAM."""
        if not hasattr(self.backbone, "feature_info"):
            return None
        try:
            last_info = self.backbone.feature_info[-1]
            module_name = last_info["module"]
            modules = dict(self.backbone.named_modules())
            layer = modules.get(module_name)
            # Inplace ops (e.g. ReLU(inplace=True)) conflict with backward hooks.
            # Walk back to the nearest preceding layer with weights (Conv/BN),
            # and disable inplace on any activations that follow it.
            if layer is not None and getattr(layer, "inplace", False):
                layer.inplace = False  # safe: only affects GradCAM pass
                all_names = list(modules.keys())
                idx = all_names.index(module_name) if module_name in all_names else -1
                for i in range(idx - 1, -1, -1):
                    candidate = modules[all_names[i]]
                    if not getattr(candidate, "inplace", False) and hasattr(candidate, "weight"):
                        return candidate
            return layer
        except Exception:
            return None
