from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import cv2

from src.new_terraces_generator.generator import build_rotated_terraced_dimer_canvas


# Alle möglichen Störungen
ALL_PERTURBATIONS = [
    "tilt",
    "oscillation",
    "xy",
    "zones",
    "amp_noise",
    "white_artifacts",
]

# Flags, die den Ordner definieren 
FOLDER_BASE_FLAGS = [
    "xy",
    "zones",
    "amp_noise",
    "white_artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STM-Terrassen-Bilder mit Dimern generieren."
    )

    parser.add_argument(
        "-n",
        "--n-per-folder",
        type=int,
        default=100,
        help="Anzahl der zu generierenden Bilder pro Ordner (Default: 100).",
    )

    parser.add_argument(
        "--base-out-dir",
        type=Path,
        default=Path("data/generated_stm"),
        help="Basis-Ausgabeverzeichnis (Default: ./data/generated_stm).",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="stm",
        help="Präfix für Ordner- und Dateinamen (Default: 'stm').",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Startindex für globale Dateibenennung (Default: 0).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Optionaler Zufallsseed für Reproduzierbarkeit.",
    )

    # Terrassen-Parameter
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="min_steps für build_rotated_terraced_dimer_canvas (Default: 1).",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="max_steps für build_rotated_terraced_dimer_canvas (Default: 5).",
    )

    # WICHTIG: default=None, damit wir erkennen, ob der User sie gesetzt hat
    parser.add_argument(
        "--enable",
        nargs="*",
        choices=ALL_PERTURBATIONS,
        default=None,
        help=(
            "Manueller Modus: explizit zu aktivierende Störungen. "
            "Wenn nicht gesetzt -> Auto-Modus (alle Varianten)."
        ),
    )

    parser.add_argument(
        "--disable",
        nargs="*",
        choices=ALL_PERTURBATIONS,
        default=None,
        help=(
            "Manueller Modus: explizit zu deaktivierende Störungen. "
            "Wenn weder --enable noch --disable gesetzt sind -> Auto-Modus."
        ),
    )

    return parser.parse_args()


def build_folder_name(prefix: str, enabled_base: set[str], include_tilt_osc: bool = True) -> str:
    """
    Baue einen Ordnernamen, der die Flags kodiert.
    enabled_base: Menge von Flags, die für den Ordner 'fest' sind.
    include_tilt_osc: Nur im manuellen Modus True; im Auto-Modus False.
    """

    def bit(flag: str) -> int:
        return 1 if flag in enabled_base else 0

    parts = [prefix]

    # xy-Drift
    parts.append(f"dx_{bit('xy')}")
    # Zonen
    parts.append(f"zones_{bit('zones')}")
    # Amplitudenrauschen
    parts.append(f"amp_{bit('amp_noise')}")
    # White artifacts
    parts.append(f"wa_{bit('white_artifacts')}")

    if include_tilt_osc:
        parts.append(f"tilt_{bit('tilt')}")
        parts.append(f"osc_{bit('oscillation')}")

    return "_".join(parts)


def auto_mode_folder_variants() -> list[set[str]]:
    """
    Auto-Modus: Erzeuge alle 2^len(FOLDER_BASE_FLAGS) Kombinationen
    der 'stabilen' Flags (xy, zones, amp_noise, white_artifacts).
    Tilt & oscillation werden später pro Bild randomisiert.
    """
    variants: list[set[str]] = []

    n = len(FOLDER_BASE_FLAGS)
    for mask in range(2**n):
        enabled = set()
        for i, flag in enumerate(FOLDER_BASE_FLAGS):
            if (mask >> i) & 1:
                enabled.add(flag)
        variants.append(enabled)

    return variants


def main() -> None:
    args = parse_args()

    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    base_out: Path = args.base_out_dir
    base_out.mkdir(parents=True, exist_ok=True)

    manual_mode = not (args.enable is None and args.disable is None)

    if manual_mode:
        # ---------- MANUELLER MODUS ----------
        enable = set(args.enable or ALL_PERTURBATIONS)
        disable = set(args.disable or [])
        enabled = enable - disable

        print("MANUELLER MODUS")
        print(f"Aktive Störungen (fix für diesen Ordner): {sorted(enabled)}")

        folder_name = build_folder_name(
            args.prefix,
            enabled_base=enabled,
            include_tilt_osc=True,
        )

        variants = [(folder_name, enabled)]

    else:
        # ---------- AUTO-MODUS ----------
        print("AUTO-MODUS: erzeuge alle Kombinationen der Basis-Flags")
        variants = []
        for enabled_base in auto_mode_folder_variants():
            folder_name = build_folder_name(
                args.prefix,
                enabled_base=enabled_base,
                include_tilt_osc=False,
            )
            variants.append((folder_name, enabled_base))

        print(f"Anzahl Ordner (Varianten): {len(variants)}")
        print(f"Basis-Flags: {FOLDER_BASE_FLAGS}")
        print(f"Bilder pro Ordner: {args.n_per_folder}")

    global_idx = args.start_index

    for folder_idx, (folder_name, enabled_base) in enumerate(variants):
        image_root     = base_out / "images"
        mask_root_sig  = base_out / "masks"      
        mask_root_row  = base_out / "masks_row" 

        img_dir      = image_root    / folder_name
        msk_sig_dir  = mask_root_sig / folder_name
        msk_row_dir  = mask_root_row / folder_name

        img_dir.mkdir(parents=True, exist_ok=True)
        msk_sig_dir.mkdir(parents=True, exist_ok=True)
        msk_row_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Ordner {folder_idx + 1}/{len(variants)} ===")
        print(f"Bilderordner: {img_dir}")
        print(f"Maskenordner (Signaturen): {msk_sig_dir}")
        print(f"Maskenordner (Dimerreihen): {msk_row_dir}")
        print(f"Basis-Flags in diesem Ordner: {sorted(enabled_base)}")

        for i in range(args.n_per_folder):
            if manual_mode:
                enabled = set(enabled_base)
            else:
                enabled = set(enabled_base)
                if random.random() < 0.5:
                    enabled.add("tilt")
                if random.random() < 0.5:
                    enabled.add("oscillation")

            img, mask_sig, mask_row = build_rotated_terraced_dimer_canvas(
                target_size=(600, 600),
                scale_factor=1.42,
                angle_deg=45.0,
                min_steps=args.min_steps,
                max_steps=args.max_steps,
                perturbations=enabled,
            )

            img_u8      = img.astype(np.uint8)
            mask_sig_u8 = mask_sig.astype(np.uint8)
            mask_row_u8 = (mask_row*1).astype(np.uint8)

            img_path     = img_dir     / f"{args.prefix}_{global_idx:06d}.png"
            msk_sig_path = msk_sig_dir / f"{args.prefix}_{global_idx:06d}_mask_sig.png"
            msk_row_path = msk_row_dir / f"{args.prefix}_{global_idx:06d}_mask_row.png"

            cv2.imwrite(str(img_path),     img_u8)
            cv2.imwrite(str(msk_sig_path), mask_sig_u8)
            cv2.imwrite(str(msk_row_path), mask_row_u8)

            global_idx += 1

    print("\nFertig.")


if __name__ == "__main__":
    main()
