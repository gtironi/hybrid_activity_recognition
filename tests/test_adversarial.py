"""Subject-adversarial GRL and HybridModel forward tests."""

import torch


def test_grl_forward_unchanged_backward_flips():
    from hybrid_activity_recognition.models.grl import GradientReversal

    grl = GradientReversal()
    x = torch.randn(4, 8, requires_grad=True)
    y = grl(x, alpha=0.5)
    assert torch.allclose(x, y)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, -0.5 * torch.ones_like(x))


def test_ganin_schedule_endpoints():
    from hybrid_activity_recognition.training.grl_schedule import ganin_grl_lambda

    assert ganin_grl_lambda(0, 10) < 0.1
    assert ganin_grl_lambda(9, 10) > 0.9


def test_hybrid_adversarial_forward():
    from hybrid_activity_recognition.models import build_hybrid_model

    b, t, k, n_cls, n_subj = 4, 75, 40, 11, 5
    model = build_hybrid_model(
        "robust",
        input_mode="hybrid",
        num_classes=n_cls,
        n_tsfel_feats=k,
        fusion_name="gated",
        num_subjects=n_subj,
        hidden_lstm=32,
    )
    x_sig = torch.randn(b, 3, t)
    x_feat = torch.randn(b, k)
    beh, subj = model(x_sig, x_feat, compute_adversarial=True, grl_alpha=0.7)
    assert beh.shape == (b, n_cls)
    assert subj.shape == (b, n_subj)

    beh_only = model(x_sig, x_feat)
    assert beh_only.shape == (b, n_cls)


def test_subject_discriminator_shape():
    from hybrid_activity_recognition.models.heads import SubjectDiscriminator

    disc = SubjectDiscriminator(in_dim=64, num_subjects=7)
    z = torch.randn(3, 64)
    out = disc(z)
    assert out.shape == (3, 7)
