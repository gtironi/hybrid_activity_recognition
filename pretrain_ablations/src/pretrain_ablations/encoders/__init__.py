from pretrain_ablations.encoders.base import Encoder
from pretrain_ablations.encoders.cnn_tfc import TFCConvEncoder
from pretrain_ablations.encoders.resnet1d import ResNet1DEncoder
from pretrain_ablations.encoders.patchtst import PatchTSTEncoder
from pretrain_ablations.encoders.patchtsmixer import PatchTSMixerEncoder


def build_encoder(cfg, in_channels: int, context_length: int) -> Encoder:
    name = cfg.name
    if name == "cnn_tfc":
        return TFCConvEncoder(
            in_channels=in_channels, kernel_size=cfg.kernel_size,
            stride=cfg.stride, cnn_dropout=cfg.cnn_dropout,
            final_out_channels=cfg.final_out_channels, d_model=cfg.d_model,
        )
    if name == "resnet1d":
        return ResNet1DEncoder(
            in_channels=in_channels, base_channels=cfg.base_channels,
            num_blocks=cfg.num_blocks, d_model=cfg.d_model,
        )
    if name == "patchtst":
        return PatchTSTEncoder(
            context_length=context_length, in_channels=in_channels,
            patch_length=cfg.patch_length, patch_stride=cfg.patch_stride,
            d_model=cfg.d_model, num_heads=cfg.num_heads,
            num_layers=cfg.num_layers, dropout=cfg.dropout,
        )
    if name == "patchtsmixer":
        return PatchTSMixerEncoder(
            context_length=context_length, in_channels=in_channels,
            patch_length=cfg.patch_length, patch_stride=cfg.patch_stride,
            d_model=cfg.d_model, num_layers=cfg.num_layers, dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown encoder name={name!r}")


__all__ = ["Encoder", "TFCConvEncoder", "ResNet1DEncoder",
           "PatchTSTEncoder", "PatchTSMixerEncoder", "build_encoder"]
