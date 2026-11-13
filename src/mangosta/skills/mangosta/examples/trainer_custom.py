"""Reference example: Custom Model, Custom BaseTrainer, and RunTracker Validation Samples."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from mangosta import (
    BaseTrainer,
    MangostaConfig,
    ModelRegistry,
    TrainerRegistry,
    load_config,
    register_model,
    register_trainer,
)


@register_model(family="vision_classifier", version="1.0.0")
class VisionClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(self.encoder(x).flatten(1))
        return out


@register_trainer(family="vision_classifier", version="1.0.0")
class VisionTrainer(BaseTrainer):
    criterion: nn.CrossEntropyLoss

    def __init__(self, model: nn.Module, config: MangostaConfig | dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(model, config, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self) -> tuple[torch.optim.Optimizer, Any]:
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.training.epochs)
        return opt, sched

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        loss: torch.Tensor = self.criterion(self.model(x), y)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor | float]:
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)
        acc = (preds.argmax(dim=-1) == y).float().mean()

        # Optional: log temporal validation sample via self.run (managed by BaseTrainer.fit)
        if self.run is not None and self.current_epoch % 5 == 0:
            self.run.log_validation_sample(
                sample_id="val_sample_0",
                prediction=preds[0].detach().cpu().numpy(),
                input_image=x[0].detach().cpu().numpy(),
                ground_truth=np.array([int(y[0].item())]),
                step=self.current_epoch,
            )

        return {"val_loss": loss, "val_acc": acc}


def run_training(config_path: str = "configs/exp_baseline.yaml", resume_from: str | None = None) -> None:
    cfg = load_config(config_path)
    model = ModelRegistry.build_from_config(cfg)
    trainer = TrainerRegistry.build_from_config(cfg, model=model)
    # trainer.fit(train_loader, val_loader, resume_from=resume_from)
    _ = trainer
