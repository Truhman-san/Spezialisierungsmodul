from __future__ import annotations
from pathlib import Path
import json


def prepare_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    return path


def save_run_config(path: Path, cfg_obj) -> None:
    cfg_json = {k: getattr(cfg_obj, k) for k in vars(cfg_obj)}
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, indent=2)