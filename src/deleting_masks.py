from pathlib import Path

# === Pfade anpassen ===
IMAGES_DIR = Path("D:\\Studium\\Semester 3\\Project Data Analytics\\stm_generation\\stm_dataset\\images")   # Oberordner
MASKS_DIR  = Path("D:\\Studium\\Semester 3\\Project Data Analytics\\stm_generation\\stm_dataset\\masks")    # Oberordner
OUTPUT_TXT = Path("deleted_masks.txt")

# 1) Alle Bilddateien rekursiv einsammeln
image_files = list(IMAGES_DIR.rglob("*.*"))  # png/jpg/tif/etc
image_stems = {p.stem for p in image_files if p.is_file()}

print(f"{len(image_stems)} Bild-Basenamen gefunden.")

# 2) Alle Masken rekursiv durchsuchen
mask_files = list(MASKS_DIR.rglob("*.*"))
deleted = []

for m in mask_files:
    if not m.is_file():
        continue

    if m.stem not in image_stems:
        deleted.append(str(m))
        m.unlink()  # --> löscht die Datei

print(f"{len(deleted)} Masken ohne zugehöriges Bild gelöscht.")

# 3) Gelöschte Masken speichern
with OUTPUT_TXT.open("w", encoding="utf-8") as f:
    for path in deleted:
        f.write(path + "\n")

print(f"Liste gespeichert in {OUTPUT_TXT.resolve()}")
