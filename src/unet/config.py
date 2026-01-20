from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class Config:
    # Data
    images_dir: str = "data/seventh_training/images"
    masks_dir: str = "data/seventh_training/masks"
    image_height: int = 600
    image_width: int = 600
    channels: int = 1 # grayscale
    num_classes: int = 4 # background + 3 defect types

    # Split
    val_fraction: float = 0.1
    test_fraction: float = 0.2
    random_seed: int = 42

    # Training
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3

    # Augmentation
    aug_rotate: bool = True
    aug_flip_h: bool = True
    aug_flip_v: bool = False
    aug_random_zoom: bool = True
    aug_zoom_range: float = 0.1 # ±10%

    # Checkpoints / logging
    output_root: str = "runs"
    run_name: str | None = None # if None -> auto timestamp

    # Compute
    use_mixed_precision: bool = False # keep false to match baseline numerics

    def output_dir(self) -> Path:
        name = self.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return Path(self.output_root) / name