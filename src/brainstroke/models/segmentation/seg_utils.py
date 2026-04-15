from .swin_unet import SwinUNet


def get_seg_output(out, model):
    if not isinstance(out, tuple):
        return out
    if isinstance(model, SwinUNet) or getattr(model, "_is_swin_unet", False):
        return out[1]
    return out[0]
