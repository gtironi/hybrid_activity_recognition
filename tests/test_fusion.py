"""Dimension sanity checks for TsfelBranch, FusionModule, and ClassificationHead."""

import torch

B = 4
D = 128


def test_mlp_tsfel_branch():
    from hybrid_activity_recognition.models.tsfel_branches import MLPTsfelBranch

    branch = MLPTsfelBranch(in_features=120, hidden_dim=D)
    x = torch.randn(B, 120)
    out = branch(x)
    assert out.shape == (B, D)
    assert branch.output_dim == D


def test_concat_fusion():
    from hybrid_activity_recognition.models.fusion import ConcatFusion

    fus = ConcatFusion(signal_dim=128, tsfel_dim=128)
    z_sig = torch.randn(B, 128)
    z_ts = torch.randn(B, 128)
    out = fus(z_sig, z_ts)
    assert out.shape == (B, 256)
    assert fus.output_dim == 256


def test_gated_fusion():
    from hybrid_activity_recognition.models.fusion import GatedFusion

    fus = GatedFusion(d_encoder=D, d_tsfel=D, d_fused=D)
    z_sig = torch.randn(B, D)
    z_ts = torch.randn(B, D)
    out = fus(z_sig, z_ts)
    assert out.shape == (B, D)
    assert fus.output_dim == D


def test_mlp_head():
    from hybrid_activity_recognition.models.heads import MLPHead

    head = MLPHead(in_dim=192, hidden_dim=128, num_classes=19)
    x = torch.randn(B, 192)
    out = head(x)
    assert out.shape == (B, 19)


def test_linear_head():
    from hybrid_activity_recognition.models.heads import LinearHead

    head = LinearHead(in_dim=256, num_classes=10)
    x = torch.randn(B, 256)
    out = head(x)
    assert out.shape == (B, 10)
