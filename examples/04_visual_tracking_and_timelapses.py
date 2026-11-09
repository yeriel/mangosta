"""Mangosta Example 04: Visual Experiment Tracking & Real-Data Timelapses.

Learn how to:
1. Track training runs with RunTracker.
2. Log scalar metrics (loss, accuracy, dice score) across epochs.
3. Log step-versioned validation samples (input image, ground truth, and evolving prediction).
4. Launch and explore the Industrial Dark visualizer: `mangosta view`.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mangosta.artifacts import save_bundle
from mangosta.core.registry import register_model
from mangosta.tracking import RunTracker


@register_model("vision_unet", "1.0.0")
class VisionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, num_classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(torch.relu(self.conv1(x)))


def generate_synthetic_image(h: int = 128, w: int = 128, pattern: str = "circle") -> np.ndarray:
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.clip(30 + 15 * np.sin(x / 10), 0, 255)
    img[:, :, 1] = np.clip(35 + 15 * np.cos(y / 10), 0, 255)
    img[:, :, 2] = np.clip(50 + 20 * np.sin((x + y) / 15), 0, 255)

    if pattern == "circle":
        mask = dist < 35
        img[mask] = [180, 220, 255]
    elif pattern == "ring":
        mask = (dist > 25) & (dist < 45)
        img[mask] = [255, 180, 100]
    return img


def generate_mask(h: int = 128, w: int = 128, radius: int = 35) -> np.ndarray:
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = (dist < radius).astype(np.uint8) * 255
    return np.stack([mask, mask, mask], axis=-1)


def generate_evolving_prediction(h: int = 128, w: int = 128, epoch: int = 0, max_epochs: int = 6, radius: int = 35) -> np.ndarray:
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    progress = (epoch + 1) / max_epochs
    noise = np.random.normal(0, (1.0 - progress) * 40, (h, w))
    pred_field = (radius - dist) * (progress * 5) + noise

    prob = 1.0 / (1.0 + np.exp(-np.clip(pred_field / 10.0, -10, 10)))
    prob_img = (prob * 255).astype(np.uint8)

    heat = np.zeros((h, w, 3), dtype=np.uint8)
    heat[:, :, 0] = np.clip(prob_img * 1.2, 0, 255).astype(np.uint8)
    heat[:, :, 1] = np.clip(prob_img * 0.7, 0, 255).astype(np.uint8)
    heat[:, :, 2] = np.clip(255 - prob_img, 0, 255).astype(np.uint8)
    return heat


def main() -> None:
    print("=" * 75)
    print("MANGOSTA TUTORIAL 04: VISUAL EXPERIMENT TRACKING & TIMELAPSES")
    print("=" * 75)

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "experiments"

        # Run 1: Baseline UNet
        print("\n[Step 1] Logging multi-epoch run with validation sample timelapses...")
        with RunTracker(experiment="medical_imaging", run_name="baseline_unet_lr1e-3", root_dir=root) as r1:
            r1.log_params({
                "learning_rate": 1e-3,
                "optimizer": "AdamW",
                "batch_size": 16,
                "backbone": "resnet18",
                "loss": "DiceBCE",
            })

            real_sample = generate_synthetic_image(pattern="circle")
            gt_sample = generate_mask(radius=35)

            for epoch in range(4):
                loss = 0.85 * math.exp(-0.45 * epoch) + 0.05
                dice = 0.40 + 0.52 * (1.0 - math.exp(-0.6 * epoch))

                r1.log_metrics({
                    "train/loss": float(loss),
                    "val/dice_score": float(dice),
                    "val/loss": float(loss + 0.04),
                }, step=epoch)

                # Log step-versioned validation prediction
                pred = generate_evolving_prediction(epoch=epoch, max_epochs=4, radius=35)
                r1.log_validation_sample(
                    sample_id="patient_01_lesion",
                    prediction=pred,
                    input_image=real_sample,
                    ground_truth=gt_sample,
                    step=epoch,
                )
                print(f"  ✓ Logged epoch {epoch + 1}/4 (train/loss: {loss:.4f}, val/dice: {dice:.4f})")

        # Run 2: Checkpoint for Bundle Inspector
        print("\n[Step 2] Packaging sample bundle for Bundle Inspector...")
        model = VisionUNet(in_channels=3, num_classes=1)
        chk_path = root / "checkpoints" / "unet_epoch4.bundle"
        chk_path.parent.mkdir(parents=True, exist_ok=True)
        save_bundle(
            output_path=chk_path,
            model=model,
            family="vision_unet",
            version="1.0.0",
            sample_input=torch.randn(1, 3, 128, 128),
            config={"architecture": "VisionUNet", "norm": "batch"},
            metrics={"val_dice": 0.92, "val_loss": 0.08},
            epoch=4,
            step=400,
        )
        print(f"  ✓ Saved bundle: {chk_path.name}")

        print("\n" + "=" * 75)
        print("SUMMARY: Synthetic visual run and bundle generated successfully.")
        print("To view experiments with the live visualizer, run in your terminal:")
        print(f"  mangosta view --dir {root}")
        print("=" * 75)


if __name__ == "__main__":
    main()
