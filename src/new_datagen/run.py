from src.new_datagen.generator import generate_synthetic_stm_sample
import matplotlib.pyplot as plt

img, mask = generate_synthetic_stm_sample(
    canvas_size=(600, 600),
    with_terraces=True,
    with_defects=False,
    rotate_45_deg=True,
)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("STM Synth")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="viridis")
plt.title("Signatur-Maske")
plt.show()
