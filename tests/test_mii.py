import torch

from miia.models.mii import MaskedInteractionInferring
from miia.models.dvae import DeterministicVisualTokenizer


def test_mii_shapes_and_masked_losses():
    model = MaskedInteractionInferring(
        image_width=12,
        text_width=8,
        hidden_dim=16,
        heads=4,
        layers=1,
        visual_vocab_size=32,
        text_vocab_size=40,
    )
    image = torch.randn(2, 4, 12)
    text = torch.randn(2, 6, 8)
    image_logits = model.image_logits(image, text, torch.zeros(2, 6, dtype=torch.bool))
    text_logits = model.text_logits(text, image)
    assert image_logits.shape == (2, 4, 32)
    assert text_logits.shape == (2, 6, 40)
    image_mask = torch.tensor([[True, False, False, True], [False, True, True, False]])
    labels = torch.randint(0, 32, (2, 4))
    loss = model.masked_loss(image_logits, labels, image_mask)
    loss.backward()
    assert torch.isfinite(loss)


def test_visual_tokenizer_grid_aligns_with_vit_patches():
    tokenizer = DeterministicVisualTokenizer()
    labels = tokenizer(torch.rand(2, 3, 128, 128), target_grid=(16, 16))
    assert labels.shape == (2, 16, 16)
    assert labels.min() >= 0
    assert labels.max() < 8192
