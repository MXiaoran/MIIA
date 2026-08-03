from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .clip_backbone import ClipBackbone, create_clip_backbone
from .dvae import create_visual_tokenizer
from .losses import bdm_loss, dcl_loss, paired_logits
from .masking import random_mask, text_token_mask
from .mii import MaskedInteractionInferring
from .queues import PairedFeatureQueue


@dataclass
class MIIAOutput:
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    image_features: torch.Tensor
    text_features: torch.Tensor
    queued_image_keys: torch.Tensor
    queued_text_keys: torch.Tensor
    dataset: str


class MIIA(nn.Module):
    """Paper-faithful MIIA reconstruction with dataset-specific paired queues."""

    def __init__(
        self,
        backbone: ClipBackbone,
        model_config: dict[str, Any],
        project_root: Path,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # Keep EMA as direct child modules to make state_dict and device transfer reliable.
        self.momentum_visual = copy.deepcopy(backbone.visual)
        self.momentum_token_embedding = copy.deepcopy(backbone.token_embedding)
        self.momentum_text_transformer = copy.deepcopy(backbone.text_transformer)
        self.momentum_text_positional_embedding = nn.Parameter(
            backbone.text_positional_embedding.detach().clone(), requires_grad=False
        )
        self.momentum_text_ln_final = copy.deepcopy(backbone.text_ln_final)
        projection = backbone.text_projection
        if isinstance(projection, nn.Linear):
            self.momentum_text_projection = copy.deepcopy(projection)
            self.register_parameter("momentum_text_projection_parameter", None)
        else:
            self.momentum_text_projection = None
            self.momentum_text_projection_parameter = nn.Parameter(projection.detach().clone(), requires_grad=False)

        self.momentum = float(model_config["dcl"]["momentum"])
        self.temperature = float(model_config["dcl"]["temperature"])
        self.image_mask_ratio = float(model_config["image_mask_ratio"])
        self.text_mask_ratio = float(model_config["text_mask_ratio"])
        self.mii_enabled = bool(model_config["mii"].get("enabled", True))
        self.dcl_enabled = bool(model_config["dcl"].get("enabled", True))
        self.bdm_enabled = bool(model_config["bdm"].get("enabled", True))
        self.loss_weights = {key: float(value) for key, value in model_config["loss_weights"].items()}

        mii_config = model_config["mii"]
        backbone.set_gradient_checkpointing(bool(mii_config.get("gradient_checkpointing", False)))
        self.mii = MaskedInteractionInferring(
            image_width=backbone.image_width,
            text_width=backbone.text_width,
            hidden_dim=int(mii_config["hidden_dim"]),
            heads=int(mii_config["heads"]),
            layers=int(mii_config["layers"]),
            visual_vocab_size=int(mii_config["visual_vocab_size"]),
            text_vocab_size=int(mii_config["text_vocab_size"]),
            checkpointing=bool(mii_config.get("gradient_checkpointing", False)),
        )
        self.visual_tokenizer = create_visual_tokenizer(model_config["dvae"], project_root)
        self.visual_tokenizer.eval()

        queue_sizes = model_config["dcl"]["queue_sizes"]
        self.queues = nn.ModuleDict({
            dataset: PairedFeatureQueue(int(size), backbone.embed_dim) for dataset, size in queue_sizes.items()
        })
        self._freeze_momentum()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> tuple["MIIA", Any]:
        image_size = int(config["data"]["image_size"])
        backbone, tokenizer = create_clip_backbone(
            config["model"]["clip_model"], config["model"]["clip_pretrained"], image_size=image_size
        )
        return cls(backbone, config["model"], Path(config["project_root"])), tokenizer

    def _freeze_momentum(self) -> None:
        modules = [
            self.momentum_visual,
            self.momentum_token_embedding,
            self.momentum_text_transformer,
            self.momentum_text_ln_final,
        ]
        if self.momentum_text_projection is not None:
            modules.append(self.momentum_text_projection)
        for module in modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def initialize_momentum(self) -> None:
        self.momentum_visual.load_state_dict(self.backbone.visual.state_dict())
        self.momentum_token_embedding.load_state_dict(self.backbone.token_embedding.state_dict())
        self.momentum_text_transformer.load_state_dict(self.backbone.text_transformer.state_dict())
        self.momentum_text_ln_final.load_state_dict(self.backbone.text_ln_final.state_dict())
        self.momentum_text_positional_embedding.copy_(self.backbone.text_positional_embedding)
        if self.momentum_text_projection is not None:
            self.momentum_text_projection.load_state_dict(self.backbone.text_projection.state_dict())
        else:
            self.momentum_text_projection_parameter.copy_(self.backbone.text_projection)
        self._freeze_momentum()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_momentum()
        self.visual_tokenizer.eval()
        return self

    def new_module_parameters(self):
        named = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name in {"backbone.image_mask_embedding", "backbone.text_mask_embedding"}:
                named.append(parameter)
                continue
            if (
                name.startswith("backbone.visual")
                or name.startswith("backbone.token_embedding")
                or name.startswith("backbone.text_")
            ):
                continue
            named.append(parameter)
        return named

    def pretrained_parameters(self):
        new_ids = {id(parameter) for parameter in self.new_module_parameters()}
        return [parameter for parameter in self.parameters() if parameter.requires_grad and id(parameter) not in new_ids]

    @torch.no_grad()
    def update_momentum(self) -> None:
        online_parameters = list(self.backbone.visual.parameters())
        momentum_parameters = list(self.momentum_visual.parameters())
        online_parameters += list(self.backbone.token_embedding.parameters())
        momentum_parameters += list(self.momentum_token_embedding.parameters())
        online_parameters += list(self.backbone.text_transformer.parameters())
        momentum_parameters += list(self.momentum_text_transformer.parameters())
        online_parameters += list(self.backbone.text_ln_final.parameters())
        momentum_parameters += list(self.momentum_text_ln_final.parameters())
        if isinstance(self.backbone.text_projection, nn.Linear):
            online_parameters += list(self.backbone.text_projection.parameters())
            momentum_parameters += list(self.momentum_text_projection.parameters())
        for online, momentum in zip(online_parameters, momentum_parameters):
            momentum.data.mul_(self.momentum).add_(online.data, alpha=1 - self.momentum)
        self.momentum_text_positional_embedding.data.mul_(self.momentum).add_(
            self.backbone.text_positional_embedding.data, alpha=1 - self.momentum
        )
        if self.momentum_text_projection_parameter is not None:
            self.momentum_text_projection_parameter.data.mul_(self.momentum).add_(
                self.backbone.text_projection.data, alpha=1 - self.momentum
            )

    @torch.no_grad()
    def _momentum_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        visual = self.momentum_visual
        value = visual.conv1(image).reshape(image.shape[0], visual.conv1.out_channels, -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(value.dtype).reshape(1, 1, -1).expand(value.shape[0], -1, -1)
        value = torch.cat([class_token, value], dim=1)
        from .clip_backbone import interpolate_visual_positional_embedding

        patch_height = image.shape[-2] // visual.conv1.kernel_size[0]
        patch_width = image.shape[-1] // visual.conv1.kernel_size[1]
        value = value + interpolate_visual_positional_embedding(
            visual.positional_embedding,
            (patch_height, patch_width),
        ).to(value.dtype)
        value = visual.ln_pre(value).permute(1, 0, 2)
        value = visual.transformer(value).permute(1, 0, 2)
        value = visual.ln_post(value)[:, 0]
        if visual.proj is not None:
            value = value @ visual.proj
        return torch.nn.functional.normalize(value, dim=-1)

    @torch.no_grad()
    def _momentum_encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        value = self.momentum_token_embedding(token_ids) + self.momentum_text_positional_embedding[: token_ids.shape[1]].to(
            self.momentum_token_embedding.weight.dtype
        )
        value = value.permute(1, 0, 2)
        try:
            value = self.momentum_text_transformer(value, attn_mask=self.backbone.text_attn_mask)
        except TypeError:
            value = self.momentum_text_transformer(value)
        value = self.momentum_text_ln_final(value.permute(1, 0, 2))
        pooled = value[torch.arange(value.shape[0], device=value.device), token_ids.argmax(dim=-1)]
        if self.momentum_text_projection is not None:
            pooled = self.momentum_text_projection(pooled)
        else:
            pooled = pooled @ self.momentum_text_projection_parameter
        return torch.nn.functional.normalize(pooled, dim=-1)

    def forward(self, batch: dict[str, Any]) -> MIIAOutput:
        image: torch.Tensor = batch["image"]
        text: torch.Tensor = batch["text"]
        dataset_names = batch["dataset"]
        dataset = dataset_names[0] if isinstance(dataset_names, (list, tuple)) else str(dataset_names)
        if any(name != dataset for name in dataset_names):
            raise ValueError("Each logical training batch must contain a single source dataset")
        if dataset not in self.queues:
            raise KeyError(f"No queue configured for dataset {dataset!r}")

        image_features, image_tokens = self.backbone.encode_image(image)
        text_features, text_tokens = self.backbone.encode_text(text)
        with torch.no_grad():
            image_keys = self._momentum_encode_image(image)
            text_keys = self._momentum_encode_text(text)

        queue = self.queues[dataset]
        logits_i2t, logits_t2i = paired_logits(
            image_features,
            text_features,
            image_keys,
            text_keys,
            queue.image.detach(),
            queue.text.detach(),
            self.temperature,
        )
        losses: dict[str, torch.Tensor] = {}
        if self.dcl_enabled:
            losses["dcl"] = dcl_loss(logits_i2t, logits_t2i)
        if self.bdm_enabled:
            losses["bdm"] = bdm_loss(logits_i2t, logits_t2i)

        if self.mii_enabled:
            image_mask = random_mask(image.shape[0], image_tokens.shape[1], self.image_mask_ratio, image.device)
            word_mask = text_token_mask(text, self.text_mask_ratio)
            _, masked_image_tokens = self.backbone.encode_image(image, patch_mask=image_mask)
            _, masked_text_tokens = self.backbone.encode_text(text, token_mask=word_mask)
            image_logits = self.mii.image_logits(masked_image_tokens, text_tokens, text.eq(0))
            text_logits = self.mii.text_logits(masked_text_tokens, image_tokens)
            with torch.no_grad():
                side = int(image_tokens.shape[1] ** 0.5)
                if side * side != image_tokens.shape[1]:
                    raise ValueError("MII currently requires a square visual patch grid")
                visual_labels = self.visual_tokenizer(
                    batch["dvae_image"].to(image.device), target_grid=(side, side)
                ).reshape(image.shape[0], -1)
            if visual_labels.shape != image_mask.shape:
                raise ValueError(f"dVAE target grid {visual_labels.shape} does not match mask grid {image_mask.shape}")
            cmii = self.mii.masked_loss(image_logits, visual_labels, image_mask)
            cmli = self.mii.masked_loss(text_logits, text, word_mask)
            losses["mii"] = cmii + cmli
            losses["cmii"] = cmii.detach()
            losses["cmli"] = cmli.detach()

        total = image_features.sum() * 0
        for name in ("mii", "dcl", "bdm"):
            if name in losses:
                total = total + self.loss_weights[name] * losses[name]
        return MIIAOutput(total, losses, image_features, text_features, image_keys, text_keys, dataset)

    @torch.no_grad()
    def enqueue(self, dataset: str, image_keys: torch.Tensor, text_keys: torch.Tensor) -> None:
        self.queues[dataset].enqueue(image_keys, text_keys)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_image(image)[0]

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_text(token_ids)[0]
