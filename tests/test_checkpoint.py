import torch

from miia.checkpoint import load_online_backbone_checkpoint
from miia.models.builders import create_tiny_miia


def test_inference_checkpoint_loads_only_online_backbone(tmp_path):
    source, _ = create_tiny_miia()
    target, _ = create_tiny_miia()
    path = tmp_path / "miia.pth"
    torch.save({"model": source.state_dict(), "global_step": 7}, path)
    payload = load_online_backbone_checkpoint(path, target.backbone)
    assert payload["global_step"] == 7
    for source_value, target_value in zip(source.backbone.state_dict().values(), target.backbone.state_dict().values()):
        assert torch.equal(source_value, target_value)
