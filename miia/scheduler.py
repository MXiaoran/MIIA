from __future__ import annotations

import math


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        warmup_start_lr: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(0, warmup_steps)
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.step_index = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._apply()

    def _factor_lr(self, base_lr: float) -> float:
        if self.warmup_steps and self.step_index < self.warmup_steps:
            progress = self.step_index / max(1, self.warmup_steps)
            return self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
        progress = (self.step_index - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def _apply(self) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._factor_lr(base_lr)

    def step(self) -> None:
        self.step_index += 1
        self._apply()

    def state_dict(self) -> dict:
        return {"step_index": self.step_index, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict) -> None:
        self.step_index = int(state["step_index"])
        self.base_lrs = list(state["base_lrs"])
        self._apply()
