"""
MSDS25044_05_task2_augmentations.py
-------------------------------------
Task 2: Understanding Augmentations (8 Marks)

Visualises 10 CIFAR-10 images in the format required by the assignment:
  Original Image | Augmented View 1 | Augmented View 2
  Original Image | Augmented View 1 | Augmented View 2
  ...

Also implements TwoViewTransform (required to be written yourself).

Output:
  results/augmentation_examples.png

Run:
  python MSDS25044_05_task2_augmentations.py --data-dir data/cifar-10-batches-py
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset_splits import get_cifar10_subset
from utils.seed import set_seed

# ── CIFAR-10 constants ────────────────────────────────────────────────────────
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
CLASSES    = ["airplane","automobile","bird","cat","deer",
              "dog","frog","horse","ship","truck"]


# ════════════════════════════════════════════════════════════════════════════
# TwoViewTransform is the core idea behind SimCLR's augmentations. It applies the same
# random transform TWICE to the same image, producing two different augmented views.
# ════════════════════════════════════════════════════════════════════════════
class TwoViewTransform:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)   # first random augmentation
        view2 = self.transform(x)   # second random augmentation (different!)
        return view1, view2


# ── SimCLR augmentation pipeline (Task 2) ─
def get_simclr_transform():
    return T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])

# ── plain transform just for showing original image ───────────────────────────
def get_plain_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


# ── unnormalise tensor → HWC numpy for matplotlib ─────────────────────────────
def to_displayable(tensor):
    mean = np.array(CIFAR_MEAN).reshape(3, 1, 1)
    std  = np.array(CIFAR_STD).reshape(3, 1, 1)
    img  = tensor.numpy() * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


# ── main visualisation ────────────────────────────────────────────────────────
def visualise_augmentations(data_dir, splits_dir, out_path, n=10, seed=2026):
    set_seed(seed)

    # Raw PIL dataset — no transform, so we can apply transforms manually
    raw_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=None
    )

    # Pick n images from the labeled split
    split_file = os.path.join(splits_dir, "train_labeled_10percent.txt")
    with open(split_file) as f:
        indices = [int(l.strip()) for l in f if l.strip()]

    # Spread picks evenly across the labeled set
    step    = max(1, len(indices) // n)
    picks   = [indices[i * step] for i in range(n)]

    plain_tf    = get_plain_transform()
    simclr_tf   = get_simclr_transform()
    two_view_tf = TwoViewTransform(simclr_tf)

    # ── draw grid: n rows × 3 columns ──────────────────────────────────────
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.3})

    for col, title in enumerate(
        ["Original Image", "Augmented View 1", "Augmented View 2"]
    ):
        axes[0][col].set_title(title, fontsize=13, fontweight="bold", pad=8)

    for row, idx in enumerate(picks):
        pil_img, label = raw_dataset[idx]

        orig        = plain_tf(pil_img)
        view1, view2 = two_view_tf(pil_img)

        for col, tensor in enumerate([orig, view1, view2]):
            axes[row][col].imshow(to_displayable(tensor))
            axes[row][col].axis("off")

        # class label on the left
        axes[row][0].set_ylabel(
            CLASSES[label], fontsize=10, rotation=0,
            labelpad=45, va="center"
        )

    fig.suptitle(
        "SimCLR Augmentation Examples\n"
        "Each row shows the same image with two independently augmented views",
        fontsize=13, fontweight="bold", y=1.01
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data/cifar-10-batches-py")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--out",        default="results")
    parser.add_argument("--n-images",   type=int, default=10)
    parser.add_argument("--seed",       type=int, default=2026)
    args = parser.parse_args()

    print("Generating augmentation visualisation...")
    visualise_augmentations(
        args.data_dir,
        args.splits_dir,
        os.path.join(args.out, "augmentation_examples.png"),
        n=args.n_images,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
