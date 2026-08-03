import torch

from miia.models.clip_backbone import interpolate_visual_positional_embedding


def test_visual_position_interpolation_keeps_cls():
    source = torch.randn(197, 768)
    result = interpolate_visual_positional_embedding(source, (16, 16))
    assert result.shape == (257, 768)
    assert torch.equal(result[0], source[0])


def test_visual_position_noop_for_same_grid():
    source = torch.randn(197, 32)
    assert interpolate_visual_positional_embedding(source, (14, 14)) is source

