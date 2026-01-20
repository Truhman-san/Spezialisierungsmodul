from pathlib import Path
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent        # <projekt>/src/real_stm
PROJECT_ROOT = ROOT_DIR.parent.parent             # <projekt>

INPUT_DIR = PROJECT_ROOT / "real_stm_raw_tifs"
OUTPUT_DIR = PROJECT_ROOT / "real_stm_converted_png"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (600, 600)

print("Using INPUT_DIR:", INPUT_DIR)
print("Exists?", INPUT_DIR.exists())
print("Found files:", list(INPUT_DIR.glob("*")))

# alle TIF-Varianten einsammeln
tif_files = list(INPUT_DIR.glob("*.tif")) + list(INPUT_DIR.glob("*.tiff"))
print("TIFs:", tif_files)

for tif_path in tif_files:
    img = Image.open(tif_path)

    # Palette / RGB → Graustufe
    if img.mode != "L":
        img = img.convert("L")

    # Resize falls nötig
    if img.size != TARGET_SIZE:
        img = img.resize(TARGET_SIZE, Image.BILINEAR)

    out_path = OUTPUT_DIR / (tif_path.stem + ".png")
    img.save(out_path)
    print("Saved:", out_path)
