"""
MSDS25044_05_task3_similarity.py
---------------------------------
Task 3: Feature Similarity Before Training 

Passes images through a RANDOM (untrained) ResNet-18 encoder.
Computes cosine similarity between:
  - Same image, two augmented views   (should be low before training)
  - Different images                  (should also be low before training)

Outputs:
  results/similarity_matrix_before_training.png

Run:
  python MSDS25044_05_task3_similarity.py --data-dir data
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.seed import set_seed

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


# ── TwoViewTransform ──────────────────────────────────────────────────────────
class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x), self.transform(x)


# ── SimCLR augmentation ───────────────────────────────────────────────────────
def get_simclr_transform():
    return T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


# ── ResNet-18 encoder (no projection head, output = 512-dim) ─────────────────
def build_encoder():
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Identity()   # remove classifier → output 512-dim features
    return model


# ── Extract features for a batch ─────────────────────────────────────────────
def extract_features(encoder, view1, view2, device):
    encoder.eval()
    with torch.no_grad():
        z1 = encoder(view1.to(device))   # (N, 512)
        z2 = encoder(view2.to(device))   # (N, 512)
    # L2 normalise
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return z1, z2


# ── Compute cosine similarity matrix (2N x 2N) ───────────────────────────────
def compute_similarity_matrix(z1, z2):
    """
    Concatenates z1 and z2 into a 2N vector and computes
    the full cosine similarity matrix of shape (2N, 2N).

    For batch of N images:
      - Indices 0..N-1   = view 1 of each image
      - Indices N..2N-1  = view 2 of each image
      - Positive pair for image i = (i, i+N)
    """
    z = torch.cat([z1, z2], dim=0)          # (2N, 512)
    sim = torch.mm(z, z.t())                # cosine sim (already L2 normalised)
    return sim.cpu().numpy()


# ── Plot similarity matrix heatmap ────────────────────────────────────────────
def plot_similarity_matrix(sim_matrix, out_path, title, n):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Mark positive pair positions
    for i in range(n):
        ax.add_patch(plt.Rectangle((i + n - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="blue", linewidth=2))
        ax.add_patch(plt.Rectangle((i - 0.5, i + n - 0.5), 1, 1,
                                    fill=False, edgecolor="blue", linewidth=2))

    ax.axhline(y=n - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.axvline(x=n - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.set_title(f"{title}\n(Blue boxes = positive pairs)", fontsize=12)
    ax.set_xlabel("2N augmented views")
    ax.set_ylabel("2N augmented views")

    # Add quadrant labels
    mid = n // 2
    ax.text(mid,       mid,       "View1\nvs\nView1", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
    ax.text(mid + n,   mid,       "View1\nvs\nView2", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
    ax.text(mid,       mid + n,   "View2\nvs\nView1", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
    ax.text(mid + n,   mid + n,   "View2\nvs\nView2", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Compute average similarities ─────────────────────────────────────────────
def compute_average_similarities(z1, z2):
    """
    Returns:
      same_view_sim    : avg cosine similarity between view1 and view2 of SAME image
      diff_image_sim   : avg cosine similarity between views of DIFFERENT images
    """
    n = z1.shape[0]

    # Same image: dot product of z1[i] and z2[i]
    same_sim = (z1 * z2).sum(dim=1).mean().item()

    # Different images: all cross-image pairs (exclude same-image pairs)
    z = torch.cat([z1, z2], dim=0)          # (2N, 512)
    sim_matrix = torch.mm(z, z.t())         # (2N, 2N)

    # Mask: exclude diagonal and positive pairs
    mask = torch.ones(2 * n, 2 * n, dtype=torch.bool)
    for i in range(2 * n):
        mask[i, i] = False                  # diagonal
    for i in range(n):
        mask[i, i + n] = False              # positive pairs (view1 -> view2)
        mask[i + n, i] = False              # positive pairs (view2 -> view1)

    diff_sim = sim_matrix[mask].mean().item()
    return same_sim, diff_sim


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--out",        default="results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed",       type=int, default=2026)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"\nDevice: {device}")

    # ── Load a small batch using split file ───────────────────────────────
    two_view_tf = TwoViewTransform(get_simclr_transform())

    # Custom dataset that returns two views
    class TwoViewDataset(torch.utils.data.Dataset):
        def __init__(self, base, transform):
            self.base = base
            self.transform = transform
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, label = self.base[idx]
            v1, v2 = self.transform(img)
            return v1, v2, label

    split_file = os.path.join(args.splits_dir, "train_labeled_10percent.txt")
    with open(split_file) as f:
        indices = [int(l.strip()) for l in f if l.strip()]
    indices = indices[:args.batch_size]   # take only batch_size images

    raw = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=None)
    subset  = Subset(raw, indices)
    dataset = TwoViewDataset(subset, two_view_tf)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    view1, view2, labels = next(iter(loader))
    print(f"Batch size: {view1.shape[0]} images")

    # ── Random encoder ────────────────────────────────────────────────────
    encoder = build_encoder().to(device)
    print("Using RANDOM (untrained) encoder")

    z1, z2 = extract_features(encoder, view1, view2, device)

    # ── Similarity matrix ─────────────────────────────────────────────────
    sim_matrix = compute_similarity_matrix(z1, z2)
    plot_similarity_matrix(
        sim_matrix,
        os.path.join(args.out, "similarity_matrix_before_training.png"),
        title="Cosine Similarity Matrix — Random Encoder (Before Training)",
        n=args.batch_size
    )

    # ── Average similarities ──────────────────────────────────────────────
    same_sim, diff_sim = compute_average_similarities(z1, z2)

    print("\n" + "="*55)
    print("  Feature Similarity — BEFORE SimCLR Training")
    print("="*55)
    print(f"  Same image, two augmented views : {same_sim:.4f}")
    print(f"  Different images                : {diff_sim:.4f}")
    print("="*55)
    print("\n  (Both values should be similar/random before training)")
    print("  (After SimCLR training, same-image similarity should increase)\n")

    # Save for report
    with open(os.path.join(args.out, "similarity_before.txt"), "w") as f:
        f.write(f"same_view_similarity_before={same_sim:.6f}\n")
        f.write(f"different_image_similarity_before={diff_sim:.6f}\n")

    # Print positive pair table for report (Task 4.2)
    print("  Positive Pair Table (for your report):")
    print(f"  {'Original Image':<15} {'View 1 Index':<15} {'View 2 Index':<15} {'Positive Pair'}")
    print("  " + "-"*55)
    for i in range(min(4, args.batch_size)):
        print(f"  {'image ' + str(i):<15} {i:<15} {i + args.batch_size:<15} yes")


if __name__ == "__main__":
    main()
