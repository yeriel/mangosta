"""Generate realistic multi-run training data and validation timelapses for mangosta view."""

import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from mangosta.tracking import RunTracker
from mangosta.artifacts import save_bundle
from mangosta.core.registry import register_model


@register_model("vision_unet", "1.0.0")
class VisionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, num_classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(torch.relu(self.conv1(x)))


def generate_synthetic_image(h=128, w=128, pattern="circle"):
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Base background texture
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


def generate_mask(h=128, w=128, radius=35):
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = (dist < radius).astype(np.uint8) * 255
    # Convert to 3 channel for RGB display
    return np.stack([mask, mask, mask], axis=-1)


def generate_evolving_prediction(h=128, w=128, epoch=0, max_epochs=6, radius=35):
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    progress = (epoch + 1) / max_epochs
    noise = np.random.normal(0, (1.0 - progress) * 40, (h, w))
    pred_field = (radius - dist) * (progress * 5) + noise

    # Sigmoid probability
    prob = 1.0 / (1.0 + np.exp(-np.clip(pred_field / 10.0, -10, 10)))
    prob_img = (prob * 255).astype(np.uint8)

    # False-color heatmap for segmentation prediction
    heat = np.zeros((h, w, 3), dtype=np.uint8)
    heat[:, :, 0] = np.clip(prob_img * 1.2, 0, 255).astype(np.uint8)  # Red/amber
    heat[:, :, 1] = np.clip(prob_img * 0.7, 0, 255).astype(np.uint8)  # Green
    heat[:, :, 2] = np.clip(255 - prob_img, 0, 255).astype(np.uint8)  # Blue background
    return heat


def main():
    root = Path("./experiments")

    # 1. Run 1: Baseline UNet
    dummy_data = Path("data/processed_scans_v1.parquet")
    dummy_data.parent.mkdir(parents=True, exist_ok=True)
    dummy_data.write_text("dummy dataset payload", encoding="utf-8")
    with RunTracker(experiment="medical_imaging", run_name="baseline_unet_lr1e-3", root_dir=root) as r1:
        r1.log_params({
            "learning_rate": 1e-3,
            "optimizer": "AdamW",
            "batch_size": 16,
            "backbone": "resnet18",
            "loss": "DiceBCE",
            "weight_decay": 1e-4,
        })
        r1.set_dataset(dummy_data)

        real_sample_1 = generate_synthetic_image(pattern="circle")
        gt_sample_1 = generate_mask(radius=35)

        real_sample_2 = generate_synthetic_image(pattern="ring")
        gt_sample_2 = generate_mask(radius=40)

        for epoch in range(6):
            loss = 0.85 * math.exp(-0.45 * epoch) + 0.05 + np.random.normal(0, 0.015)
            dice = 0.40 + 0.52 * (1.0 - math.exp(-0.6 * epoch)) + np.random.normal(0, 0.01)
            val_loss = loss + 0.04 + np.random.normal(0, 0.01)

            r1.log_metrics({
                "train/loss": max(0.01, float(loss)),
                "val/dice_score": min(0.99, float(dice)),
                "val/loss": max(0.02, float(val_loss)),
            }, step=epoch)

            # Log validation timelapse on real data
            pred1 = generate_evolving_prediction(epoch=epoch, max_epochs=6, radius=35)
            r1.log_validation_sample(
                sample_id="patient_01_lesion",
                prediction=pred1,
                input_image=real_sample_1,
                ground_truth=gt_sample_1,
                step=epoch,
            )

            pred2 = generate_evolving_prediction(epoch=epoch, max_epochs=6, radius=40)
            r1.log_validation_sample(
                sample_id="patient_02_tissue",
                prediction=pred2,
                input_image=real_sample_2,
                ground_truth=gt_sample_2,
                step=epoch,
            )

    # 2. Run 2: Cosine Annealing + Higher LR
    with RunTracker(experiment="medical_imaging", run_name="cosine_lr5e-3", root_dir=root) as r2:
        r2.log_params({
            "learning_rate": 5e-3,
            "optimizer": "AdamW",
            "batch_size": 16,
            "backbone": "resnet34",
            "loss": "FocalDice",
            "weight_decay": 1e-4,
        })

        for epoch in range(6):
            loss = 0.90 * math.exp(-0.55 * epoch) + 0.03 + np.random.normal(0, 0.02)
            dice = 0.35 + 0.58 * (1.0 - math.exp(-0.7 * epoch)) + np.random.normal(0, 0.01)
            val_loss = loss + 0.03 + np.random.normal(0, 0.015)

            r2.log_metrics({
                "train/loss": max(0.01, float(loss)),
                "val/dice_score": min(0.99, float(dice)),
                "val/loss": max(0.02, float(val_loss)),
            }, step=epoch)

    # 3. Save a checkpoint bundle for the bundle inspector
    model = VisionUNet(in_channels=3, num_classes=1)
    chk_path = root / "checkpoints" / "unet_epoch5.bundle"
    save_bundle(
        output_path=chk_path,
        model=model,
        family="vision_unet",
        version="1.0.0",
        sample_input=torch.randn(1, 3, 128, 128),
        config={
            "architecture": "VisionUNet",
            "in_channels": 3,
            "num_classes": 1,
            "backbone": "resnet34",
            "crop_size": [128, 128],
            "norm": "batch",
        },
        metrics={"val_dice": 0.932, "val_loss": 0.071},
        epoch=5,
        step=1200,
    )

    print("Demo experiments and bundles created successfully in ./experiments")


if __name__ == "__main__":
    main()
