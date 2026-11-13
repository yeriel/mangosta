"""White-box PyTorch training engine with native Mangosta MLOps automation."""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mangosta.artifacts.bundle import save_bundle
from mangosta.artifacts.loader import resume_training
from mangosta.core.config import MangostaConfig, load_config, seed_everything
from mangosta.tracking.tracker import RunTracker

_STEP_SCHEDULER_TYPES = (
    torch.optim.lr_scheduler.OneCycleLR,
    torch.optim.lr_scheduler.CyclicLR,
)


def _resolve_device(requested: str | torch.device) -> torch.device:
    """Resolve 'auto', 'cuda', 'mps', or 'cpu' safely with CPU fallback."""
    if isinstance(requested, torch.device):
        return requested
    req = requested.strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "Requested device 'cuda' is not available; falling back to 'cpu'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return torch.device(req)


class BaseTrainer:
    """Minimalist, white-box PyTorch trainer that automates Mangosta MLOps invariants."""

    def __init__(
        self,
        model: nn.Module,
        config: MangostaConfig | dict[str, Any] | None = None,
        criterion: nn.Module | Callable[[Any, Any], torch.Tensor] | None = None,
        device: str | torch.device | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        self.cfg: MangostaConfig = load_config(config or {})
        seed_everything(self.cfg.experiment.seed)

        self.device: torch.device = _resolve_device(
            device if device is not None else self.cfg.experiment.device
        )
        self.model: nn.Module = model.to(self.device)
        self.criterion = criterion

        # Model metadata for consistent bundle manifests
        self.family: str = (
            self.cfg.model.family
            or getattr(model, "_mangosta_family", None)
            or getattr(model.__class__, "_mangosta_family", None)
            or model.__class__.__name__.lower()
        )
        self.version: str = (
            self.cfg.model.version
            if self.cfg.model.family
            else (
                getattr(model, "_mangosta_version", None)
                or getattr(model.__class__, "_mangosta_version", None)
                or self.cfg.model.version
            )
        )
        self.init_args: dict[str, Any] = dict(self.cfg.model.init_args)

        # AMP & gradient clipping configuration
        self.use_amp: bool = bool(self.cfg.training.amp) and self.device.type in ("cuda", "cpu")
        self.grad_clip_norm: float | None = self.cfg.training.grad_clip_norm
        self.scaler: torch.amp.GradScaler = scaler or torch.amp.GradScaler(
            self.device.type, enabled=self.use_amp
        )

        # Optimizer & scheduler setup
        self._step_scheduler: bool = False
        if optimizer is not None:
            self.optimizer = optimizer
            self.scheduler = scheduler
        else:
            opt_cfg = self.configure_optimizers()
            if isinstance(opt_cfg, tuple):
                self.optimizer, self.scheduler = opt_cfg[0], opt_cfg[1]
                if len(opt_cfg) > 2 and opt_cfg[2] == "step":
                    self._step_scheduler = True
            else:
                self.optimizer = opt_cfg
                self.scheduler = scheduler

        if self.scheduler is not None and isinstance(self.scheduler, _STEP_SCHEDULER_TYPES):
            self._step_scheduler = True

        self.run: RunTracker | None = None
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.train_loader: DataLoader[Any] | None = None
        self.val_loader: DataLoader[Any] | None = None

    # ------------------------------------------------------------------
    # Progressive White-Box User Interface
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:
        """Return an Optimizer or (Optimizer, Scheduler). Defaults to AdamW."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

    def training_step(self, batch: Any) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute loss tensor(s) for a single batch. Override for custom models or multi-head."""
        if self.criterion is None:
            raise NotImplementedError(
                "Subclasses of BaseTrainer must implement `training_step(self, batch)` "
                "or pass `criterion=` to BaseTrainer.__init__."
            )
        x, y = batch
        preds = self.model(x)
        return self.criterion(preds, y)

    def validation_step(self, batch: Any) -> torch.Tensor | dict[str, torch.Tensor | float] | None:
        """Compute validation metrics for a single batch (optional)."""
        if self.criterion is None:
            return None
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)
        return {"val_loss": loss}

    def optimize_step(self, batch: Any) -> dict[str, float]:
        """Execute forward, backward, clipping, and optimizer step for a single batch.

        Override this escape hatch for GANs, gradient accumulation, or custom optimization.
        """
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            step_output = self.training_step(batch)
            if isinstance(step_output, dict):
                if "loss" in step_output:
                    loss = step_output["loss"]
                else:
                    loss = sum(step_output.values())  # type: ignore[assignment]
            else:
                loss = step_output

        self.scaler.scale(loss).backward()

        if self.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None and self._step_scheduler:
            self.scheduler.step()

        return self._extract_metrics(step_output, default_key="loss", extra_total=loss)

    def extract_sample_input(self, batch: Any) -> Any:
        """Extract representative model input from a batch for ModelSchema inference."""
        if isinstance(batch, (tuple, list)):
            return batch[0]
        if isinstance(batch, Mapping):
            for key in ("input", "inputs", "image", "images", "x"):
                if key in batch:
                    return batch[key]
        return batch

    def move_to_device(self, data: Any) -> Any:
        """Recursively move tensors in batches to self.device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, tuple):
            return tuple(self.move_to_device(x) for x in data)
        if isinstance(data, list):
            return [self.move_to_device(x) for x in data]
        if isinstance(data, dict):
            return {k: self.move_to_device(v) for k, v in data.items()}
        return data

    # ------------------------------------------------------------------
    # Orchestration & MLOps Automation
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None = None,
        epochs: int | None = None,
        resume_from: str | Path | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        experiments_dir: str | Path = "experiments",
        sample_input: Any | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run the full training lifecycle with atomic tracking, bundles, and resumption."""
        self.train_loader = train_loader
        self.val_loader = val_loader

        total_epochs = epochs if epochs is not None else self.cfg.training.epochs
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 1
        if resume_from is not None:
            state = resume_training(
                bundle_path=resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler if self.use_amp else None,
                restore_rng=True,
                map_location=self.device,
            )
            start_epoch = state.epoch + 1
            self.global_step = state.step
            if verbose:
                print(f"Resumed deterministically from '{resume_from}' (starting at epoch {start_epoch})")

        if verbose:
            print(self.cfg.format_summary(resolved_device=str(self.device)))

        inferred_sample = sample_input
        best_metric: float | None = None
        epochs_no_improve = 0
        history: list[dict[str, float]] = []

        monitor_key = self.cfg.training.monitor
        monitor_mode = self.cfg.training.monitor_mode
        patience = self.cfg.training.patience

        with RunTracker(experiment=self.cfg.experiment.name, root_dir=experiments_dir) as run:
            self.run = run
            run.log_params(self.cfg.to_dict())

            for epoch in range(start_epoch, total_epochs + 1):
                self.current_epoch = epoch

                # 1. Train epoch
                self.model.train()
                train_sums: dict[str, float] = defaultdict(float)
                train_batches = 0

                for raw_batch in train_loader:
                    batch = self.move_to_device(raw_batch)
                    if inferred_sample is None:
                        inferred_sample = self._safe_cpu_sample(self.extract_sample_input(batch))

                    batch_metrics = self.optimize_step(batch)
                    for k, v in batch_metrics.items():
                        train_sums[k] += v
                    train_batches += 1
                    self.global_step += 1

                epoch_metrics: dict[str, float] = {
                    f"train_{k}" if not k.startswith("train_") else k: v / max(1, train_batches)
                    for k, v in train_sums.items()
                }
                # Expose canonical 'train_loss' alias if 'loss' was returned
                if "train_loss" not in epoch_metrics and "loss" in train_sums:
                    epoch_metrics["train_loss"] = train_sums["loss"] / max(1, train_batches)

                # 2. Validation epoch (optional)
                if val_loader is not None:
                    val_metrics = self._run_val_epoch(val_loader)
                    epoch_metrics.update(val_metrics)

                # 3. Epoch-level scheduler step
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                epoch_metrics["lr"] = current_lr

                if self.scheduler is not None and not self._step_scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        mon_val = epoch_metrics.get(monitor_key, epoch_metrics.get("train_loss", 0.0))
                        self.scheduler.step(mon_val)
                    else:
                        self.scheduler.step()

                # 4. Atomic metric logging
                run.log_metrics(epoch_metrics, step=epoch)
                history.append(epoch_metrics)

                if verbose:
                    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items())
                    print(f"Epoch {epoch}/{total_epochs} | {metrics_str}")

                # 5. Checkpoint bundle (crash-consistent + RNG)
                self._save_bundle_safe(
                    output_path=ckpt_dir / "model_checkpoint.bundle",
                    sample_input=inferred_sample,
                    epoch=epoch,
                    step=self.global_step,
                    metrics=epoch_metrics,
                    include_train_state=True,
                )

                # 6. Best model & EarlyStopping check
                target_key = monitor_key if monitor_key in epoch_metrics else "train_loss"
                if target_key in epoch_metrics:
                    current_val = epoch_metrics[target_key]
                    is_better = (
                        best_metric is None
                        or (monitor_mode == "min" and current_val < best_metric)
                        or (monitor_mode == "max" and current_val > best_metric)
                    )
                    if is_better:
                        best_metric = current_val
                        epochs_no_improve = 0
                        self._save_bundle_safe(
                            output_path=ckpt_dir / "model_best.bundle",
                            sample_input=inferred_sample,
                            epoch=epoch,
                            step=self.global_step,
                            metrics=epoch_metrics,
                            include_train_state=False,
                        )
                    else:
                        epochs_no_improve += 1
                        if patience is not None and epochs_no_improve >= patience:
                            if verbose:
                                print(
                                    f"EarlyStopping triggered at epoch {epoch} "
                                    f"(no improvement in '{target_key}' for {patience} epochs)."
                                )
                            break

            # 7. Final production bundle (lightweight, zero optimizer)
            final_path = self._save_bundle_safe(
                output_path=ckpt_dir / "model_final.bundle",
                sample_input=inferred_sample,
                epoch=self.current_epoch,
                step=self.global_step,
                metrics=history[-1] if history else {},
                include_train_state=False,
            )
            self.run = None

        return {
            "final_bundle": str(final_path),
            "best_metric": best_metric,
            "epochs_completed": self.current_epoch,
            "history": history,
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _run_val_epoch(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        self.model.eval()
        val_sums: dict[str, float] = defaultdict(float)
        val_batches = 0

        with torch.no_grad():
            for raw_batch in val_loader:
                batch = self.move_to_device(raw_batch)
                out = self.validation_step(batch)
                if out is None:
                    return {}
                step_metrics = self._extract_metrics(out, default_key="val_loss")
                for k, v in step_metrics.items():
                    val_sums[k] += v
                val_batches += 1

        return {k: v / max(1, val_batches) for k, v in val_sums.items()}

    def _save_bundle_safe(
        self,
        output_path: Path,
        sample_input: Any,
        epoch: int,
        step: int,
        metrics: dict[str, float],
        include_train_state: bool,
    ) -> Path:
        """Save bundle aligned with model family/version/init_args and CPU-safe schema."""
        was_training = self.model.training
        orig_device = self.device
        # Temporarily move sample_input to model's device for infer_schema if needed
        schema_sample = self.move_to_device(sample_input) if sample_input is not None else None
        try:
            return save_bundle(
                output_path=output_path,
                model=self.model,
                family=self.family,
                version=self.version,
                init_args=self.init_args,
                sample_input=schema_sample,
                optimizer=self.optimizer if include_train_state else None,
                scheduler=self.scheduler if include_train_state else None,
                scaler=self.scaler if (include_train_state and self.use_amp) else None,
                epoch=epoch,
                step=step,
                metrics=metrics,
                config=self.cfg.to_dict(),
            )
        finally:
            self.model.to(orig_device)
            self.model.train(was_training)

    @staticmethod
    def _safe_cpu_sample(sample: Any) -> Any:
        if isinstance(sample, torch.Tensor):
            return sample.detach().cpu()
        if isinstance(sample, tuple):
            return tuple(BaseTrainer._safe_cpu_sample(x) for x in sample)
        if isinstance(sample, list):
            return [BaseTrainer._safe_cpu_sample(x) for x in sample]
        if isinstance(sample, dict):
            return {k: BaseTrainer._safe_cpu_sample(v) for k, v in sample.items()}
        return sample

    @staticmethod
    def _extract_metrics(
        output: Any,
        default_key: str = "loss",
        extra_total: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if isinstance(output, dict):
            result: dict[str, float] = {}
            for k, v in output.items():
                result[k] = float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
            if "loss" not in result and extra_total is not None:
                result["loss"] = float(extra_total.detach().item())
            return result
        if isinstance(output, torch.Tensor):
            return {default_key: float(output.detach().item())}
        return {default_key: float(output)}
