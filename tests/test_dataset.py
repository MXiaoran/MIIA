import torch

from miia.data.dataset import CLIP_MEAN, CLIP_STD, build_dvae_transform


def test_dvae_pixels_are_recovered_from_augmented_clip_tensor():
    rgb = torch.rand(3, 32, 32)
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    normalized = (rgb - mean) / std
    recovered = build_dvae_transform(32)(normalized)
    assert torch.allclose(recovered, rgb, atol=2e-5)
