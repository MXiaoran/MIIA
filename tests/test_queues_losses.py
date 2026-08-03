import torch
from torch.nn import functional as F

from miia.models.losses import bdm_loss, dcl_loss, paired_logits
from miia.models.queues import PairedFeatureQueue
from miia.models.builders import create_tiny_miia


def test_queue_wraparound_preserves_pairs():
    queue = PairedFeatureQueue(size=4, dim=2)
    image = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    text = image.clone()
    queue.enqueue(image, text)
    queue.enqueue(torch.tensor([[-1.0, 0.0], [0.0, -1.0]]), torch.tensor([[-1.0, 0.0], [0.0, -1.0]]))
    assert int(queue.pointer) == 1
    assert torch.allclose(queue.image, queue.text)


def test_dcl_and_bdm_are_finite():
    image_query = F.normalize(torch.randn(3, 8), dim=-1)
    text_query = F.normalize(torch.randn(3, 8), dim=-1)
    image_key = F.normalize(torch.randn(3, 8), dim=-1)
    text_key = F.normalize(torch.randn(3, 8), dim=-1)
    image_queue = F.normalize(torch.randn(5, 8), dim=-1)
    text_queue = F.normalize(torch.randn(5, 8), dim=-1)
    logits = paired_logits(image_query, text_query, image_key, text_key, image_queue, text_queue, 0.02)
    assert torch.isfinite(dcl_loss(*logits))
    assert torch.isfinite(bdm_loss(*logits))


def test_bdm_is_zero_for_equal_distributions():
    logits = torch.randn(4, 7)
    assert bdm_loss(logits, logits).abs() < 1e-6


def test_ema_update_moves_toward_online_parameters():
    model, _ = create_tiny_miia()
    online = next(model.backbone.visual.parameters())
    momentum = next(model.momentum_visual.parameters())
    before = momentum.detach().clone()
    with torch.no_grad():
        online.add_(1.0)
    model.update_momentum()
    expected = before * model.momentum + online.detach() * (1 - model.momentum)
    assert torch.allclose(momentum, expected)


def test_mask_embeddings_use_new_module_learning_rate_group():
    model, _ = create_tiny_miia()
    new_ids = {id(parameter) for parameter in model.new_module_parameters()}
    assert id(model.backbone.image_mask_embedding) in new_ids
    assert id(model.backbone.text_mask_embedding) in new_ids
