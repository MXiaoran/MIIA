from __future__ import annotations

import torch


def random_mask(batch: int, length: int, ratio: float, device: torch.device) -> torch.Tensor:
    if not 0 <= ratio <= 1:
        raise ValueError(f"Mask ratio must be within [0, 1], got {ratio}")
    count = min(length, max(0, round(length * ratio)))
    mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    if count:
        noise = torch.rand(batch, length, device=device)
        positions = noise.topk(count, dim=1, largest=False).indices
        mask.scatter_(1, positions, True)
    return mask


def text_token_mask(
    token_ids: torch.Tensor,
    ratio: float,
    sot_id: int = 49406,
    eot_id: int = 49407,
    pad_id: int = 0,
) -> torch.Tensor:
    # OpenAI CLIP assigns SOT/EOT as the two highest vocabulary ids.
    # Tiny/test tokenizers use smaller vocabularies, so infer them when the
    # canonical ids are absent.
    maximum = int(token_ids.max().item()) if token_ids.numel() else 0
    if maximum < sot_id:
        eot_id = maximum
        sot_candidates = token_ids[:, 0].unique()
        sot_id = int(sot_candidates[0].item()) if len(sot_candidates) == 1 else maximum - 1
    eligible = (token_ids != pad_id) & (token_ids != sot_id) & (token_ids != eot_id)
    mask = torch.zeros_like(eligible)
    for row in range(token_ids.shape[0]):
        positions = eligible[row].nonzero(as_tuple=False).flatten()
        count = min(len(positions), max(0, round(len(positions) * ratio)))
        if count:
            selected = positions[torch.randperm(len(positions), device=positions.device)[:count]]
            mask[row, selected] = True
    return mask
