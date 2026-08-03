import torch

from miia.models.masking import random_mask, text_token_mask


def test_random_mask_has_exact_count():
    mask = random_mask(batch=3, length=256, ratio=0.4, device=torch.device("cpu"))
    assert mask.shape == (3, 256)
    assert mask.sum(dim=1).tolist() == [102, 102, 102]


def test_text_mask_never_touches_special_tokens():
    tokens = torch.tensor([
        [49406, 12, 13, 14, 49407, 0, 0],
        [49406, 50, 51, 52, 53, 49407, 0],
    ])
    mask = text_token_mask(tokens, ratio=1.0)
    assert not mask[:, 0].any()
    assert not mask[tokens.eq(49407)].any()
    assert not mask[tokens.eq(0)].any()
    assert mask[0, 1:4].all()

