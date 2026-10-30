# Generates the pipeline figures used in the README.
# Run from the repository root: python docs/make_figures.py
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw

from data_generator import CrossSectionDataBase, get_bezier_curve, get_random_points

OUT = os.path.join("docs", "figures")
os.makedirs(OUT, exist_ok=True)

random.seed(4)
np.random.seed(4)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


# 1. Bezier curve through random points
points = get_random_points(n=7, scale=1)
x, y, _ = get_bezier_curve(points, rad=0.3, edgy=0.2)

fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=120)
for a in ax:
    a.scatter(points[:, 0], points[:, 1], color="black", zorder=3)
    a.set_aspect("equal")
    a.set_xlim(-0.1, 1.1)
    a.set_ylim(-0.1, 1.1)
ax[0].set_title("get_random_points")
ax[1].plot(x, y, color="tab:blue")
ax[1].fill(x, y, color="tab:blue", alpha=0.15)
ax[1].set_title("get_bezier_curve")
save(fig, "01_bezier_curve.png")

# 2. Cross-section image -> discretized array
fine, coarse = 400, 24
img = Image.new("L", (fine, fine), 255)
ImageDraw.Draw(img).polygon(list(zip(x * 0.8 * fine + 0.1 * fine, y * 0.8 * fine + 0.1 * fine)), fill=0)

db = CrossSectionDataBase((fine, fine), L_pix=1)
db.generate_input(img_list=[img])

small = db.resize_IO(new_size=(coarse, coarse), numint=True)
arr = small.input_data[0, :, :, 0]

fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
ax[0].imshow(img, cmap="gray")
ax[0].set_title(f"PIL image ({fine}x{fine})")
ax[1].imshow(db.input_data[0, :, :, 0], cmap="binary", interpolation="none")
ax[1].set_title(f"input_data ({fine}x{fine})")
ax[2].imshow(arr, cmap="binary", interpolation="none")
for r in range(coarse):
    for c in range(coarse):
        ax[2].text(c, r, str(arr[r, c]), ha="center", va="center", fontsize=5,
                   color="white" if arr[r, c] else "gray")
ax[2].set_title(f"resize_IO -> ({coarse}x{coarse})")
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
save(fig, "02_discretization.png")

# 3. Random training samples
samples = CrossSectionDataBase((150, 150))
samples.generate(8, ["rectangle", "ellipse", "bezier", "bezier"])

fig, ax = plt.subplots(2, 4, figsize=(10, 5.4), dpi=120)
for a, inp, out in zip(ax.flat, samples.input_data, samples.output_data):
    a.imshow(inp[:, :, 0], cmap="binary", interpolation="none")
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel(f"I_x={out[0]:.2e}\nI_y={out[1]:.2e}\nA={out[3]:.0f}", fontsize=7)
fig.tight_layout()
save(fig, "03_training_samples.png")

# 4. CNN architecture (same in 01 and 02 train.py)
layers = [
    ("Input", "w x h x 1"),
    ("Conv2D", "32, 3x3, relu"),
    ("MaxPooling2D", "2x2"),
    ("Flatten", ""),
    ("Dense", "128, relu"),
    ("Dense", "4"),
]
fig, ax = plt.subplots(figsize=(12, 1.8), dpi=120)
for i, (name, info) in enumerate(layers):
    ax.add_patch(FancyBboxPatch((i * 2, 0), 1.6, 1, boxstyle="round,pad=0.05",
                                fc="#dbe9f6", ec="tab:blue"))
    ax.text(i * 2 + 0.8, 0.62, name, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(i * 2 + 0.8, 0.3, info, ha="center", va="center", fontsize=8)
    if i:
        ax.annotate("", (i * 2 - 0.05, 0.5), (i * 2 - 0.35, 0.5), arrowprops=dict(arrowstyle="->"))
ax.text(len(layers) * 2 - 0.2, 0.5, "[I_x, I_y, I_xy, A]", va="center", fontsize=9)
ax.set_xlim(-0.2, len(layers) * 2 + 2.2)
ax.set_ylim(-0.2, 1.2)
ax.axis("off")
save(fig, "04_cnn_architecture.png")

print("figures written to", OUT)
