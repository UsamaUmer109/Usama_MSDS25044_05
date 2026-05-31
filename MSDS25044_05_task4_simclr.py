"""
MSDS25044_05_task4_simclr.py
------------------------------
Task 4 + Task 5: SimCLR Implementation and Pretraining

Implements:
  - ResNet-18 encoder modified for CIFAR-10
  - Projection head: Linear(512->256) -> ReLU -> Linear(256->128)
  - Cosine similarity matrix (2N x 2N)
  - NT-Xent contrastive loss (implemented from scratch)
  - SimCLR pretraining loop (NO labels used)

Outputs:
  graphs/simclr_pretraining_loss.png
  results/similarity_matrix_after_training.png
  models/simclr_encoder.pt  (upload to Google Drive)

Run (quick 1-epoch test):
  python MSDS25044_05_task4_simclr.py --data-dir data --epochs 1

Full run (50 epochs):
  python MSDS25044_05_task4_simclr.py --data-dir data --epochs 50
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.seed import set_seed

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


# ══════════════════════════════════════════════════════════════════════════════
# TwoViewTransform
# ══════════════════════════════════════════════════════════════════════════════
class TwoViewTransform:
    """Returns two independently augmented views of the same image."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# ── SimCLR augmentation pipeline ─────────────────────────────────────────────
def get_simclr_transform():
    return T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Encoder — ResNet-18 modified for CIFAR-10
# ══════════════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    """
    ResNet-18 modified for 32x32 CIFAR-10 images.
    Changes from standard ResNet-18:
      - conv1: 7x7 stride 2 → 3x3 stride 1
      - maxpool removed
      - fc removed (output = 512-dim feature vector)
    """
    def __init__(self):
        super().__init__()
        base        = models.resnet18(weights=None)
        base.conv1  = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        # Remove final fc layer — keep everything up to avgpool
        self.encoder = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        h = self.encoder(x)          # (N, 512, 1, 1)
        h = h.view(h.size(0), -1)    # (N, 512)
        return h


# ══════════════════════════════════════════════════════════════════════════════
# Projection Head — used ONLY during SimCLR pretraining
# ══════════════════════════════════════════════════════════════════════════════
class ProjectionHead(nn.Module):
    """
    Maps 512-dim encoder output to 128-dim projection space.
    Architecture: Linear(512->256) -> ReLU -> Linear(256->128)
    Used only during contrastive pretraining, discarded for downstream tasks.
    """
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h):
        return self.net(h)


# ══════════════════════════════════════════════════════════════════════════════
# SimCLR Model — Encoder + Projection Head combined
# ══════════════════════════════════════════════════════════════════════════════
class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = Encoder()
        self.projector  = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)     # 512-dim representation
        z = self.projector(h)   # 128-dim projection
        return h, z


# ══════════════════════════════════════════════════════════════════════════════
# NT-Xent Loss — implemented from scratch (no library allowed)
# ══════════════════════════════════════════════════════════════════════════════
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    For a batch of N images with two views each (2N total):
      - Positive pair for image i: (z_i, z_{i+N})
      - All other 2N-2 pairs are negatives

    Formula for one positive pair (i, j):
      loss(i,j) = -log [ exp(sim(z_i, z_j) / tau) /
                         sum_{k != i} exp(sim(z_i, z_k) / tau) ]

    Final loss = mean over all 2N positive pairs.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.tau = temperature

    def forward(self, z1, z2):
        """
        z1, z2: (N, 128) — L2 normalised projections
        """
        N      = z1.shape[0]
        device = z1.device

        # L2 normalise
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate: first N rows = view1, last N rows = view2
        z = torch.cat([z1, z2], dim=0)          # (2N, 128)

        # Full cosine similarity matrix (2N x 2N)
        sim = torch.mm(z, z.t()) / self.tau      # (2N, 2N)

        # Mask out the diagonal (similarity of a vector with itself)
        mask_diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_diag, float("-inf"))

        # Positive pair indices:
        #   for view1[i] (row i),     positive is view2[i] (col i+N)
        #   for view2[i] (row i+N),   positive is view1[i] (col i)
        pos_indices = torch.cat([
            torch.arange(N, 2 * N, device=device),   # positives for rows 0..N-1
            torch.arange(0, N,     device=device),   # positives for rows N..2N-1
        ])                                            # shape: (2N,)

        # Cross entropy: treats positive as the "correct class"
        loss = F.cross_entropy(sim, pos_indices)
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# Dataset that returns two views (labels ignored during pretraining)
# ══════════════════════════════════════════════════════════════════════════════
class TwoViewDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base      = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label  = self.base[idx]
        v1, v2      = self.transform(img)
        return v1, v2, label   # label returned but NOT used during pretraining


# ══════════════════════════════════════════════════════════════════════════════
# Similarity matrix utilities
# ══════════════════════════════════════════════════════════════════════════════
def get_features(encoder, view1, view2, device):
    encoder.eval()
    with torch.no_grad():
        z1 = F.normalize(encoder(view1.to(device)), dim=1)
        z2 = F.normalize(encoder(view2.to(device)), dim=1)
    return z1, z2

def plot_similarity_matrix(sim_np, out_path, title, n):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_np, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i+n-0.5, i-0.5), 1, 1,
                                    fill=False, edgecolor="blue", linewidth=2))
        ax.add_patch(plt.Rectangle((i-0.5, i+n-0.5), 1, 1,
                                    fill=False, edgecolor="blue", linewidth=2))
    ax.axhline(y=n-0.5, color="white", linewidth=1.5, linestyle="--")
    ax.axvline(x=n-0.5, color="white", linewidth=1.5, linestyle="--")
    ax.set_title(f"{title}\nBlue boxes = positive pairs", fontsize=11)
    ax.set_xlabel("2N views"); ax.set_ylabel("2N views")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

def compute_avg_similarities(z1, z2):
    n        = z1.shape[0]
    same_sim = (z1 * z2).sum(dim=1).mean().item()
    z        = torch.cat([z1, z2], dim=0)
    sim      = torch.mm(z, z.t())
    mask     = torch.ones(2*n, 2*n, dtype=torch.bool, device=z1.device)
    for i in range(2*n): mask[i,i] = False
    for i in range(n):
        mask[i, i+n] = False
        mask[i+n, i] = False
    diff_sim = sim[mask].mean().item()
    return same_sim, diff_sim


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════
def train_simclr(model, loader, optimizer, criterion, device, epochs,
                 out_graphs, out_results, models_dir):
    losses = []
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for view1, view2, _ in tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs}",
                                     leave=False):
            view1, view2 = view1.to(device), view2.to(device)

            _, z1 = model(view1)
            _, z2 = model(view2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            ckpt = os.path.join(models_dir, f"simclr_epoch{epoch}.pt")
            torch.save(model.encoder.state_dict(), ckpt)

    # Save final encoder
    final_ckpt = os.path.join(models_dir, "simclr_encoder.pt")
    torch.save(model.encoder.state_dict(), final_ckpt)
    print(f"\n  Encoder saved: {final_ckpt}")

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), losses, color="steelblue", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("NT-Xent Loss")
    plt.title("SimCLR Pretraining Loss")
    plt.grid(alpha=0.3); plt.tight_layout()
    loss_path = os.path.join(out_graphs, "simclr_pretraining_loss.png")
    plt.savefig(loss_path, dpi=150); plt.close()
    print(f"  Saved: {loss_path}")

    return losses


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--out-results",default="results")
    parser.add_argument("--out-graphs", default="graphs")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--temperature",type=float, default=0.5)
    parser.add_argument("--seed",       type=int,   default=2026)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_results, exist_ok=True)
    os.makedirs(args.out_graphs,  exist_ok=True)
    os.makedirs(args.models_dir,  exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"\nDevice      : {device}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Temperature : {args.temperature}\n")

    # ── Load unlabeled data (NO labels used) ──────────────────────────────
    split_file = os.path.join(args.splits_dir, "train_ssl_unlabeled.txt")
    with open(split_file) as f:
        indices = [int(l.strip()) for l in f if l.strip()]

    raw     = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=None)
    subset  = Subset(raw, indices)
    dataset = TwoViewDataset(subset, TwoViewTransform(get_simclr_transform()))
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    print(f"Unlabeled training images : {len(subset):,}")
    print(f"Batches per epoch         : {len(loader)}\n")

    # ── Build model ───────────────────────────────────────────────────────
    model     = SimCLR().to(device)
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Similarity BEFORE training ────────────────────────────────────────
    print("Computing similarity BEFORE training...")
    sample_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    v1_sample, v2_sample, _ = next(iter(sample_loader))
    z1_before, z2_before = get_features(model.encoder, v1_sample, v2_sample, device)
    same_before, diff_before = compute_avg_similarities(z1_before, z2_before)

    sim_before = torch.mm(
        torch.cat([z1_before, z2_before], dim=0),
        torch.cat([z1_before, z2_before], dim=0).t()
    ).cpu().numpy()
    plot_similarity_matrix(
        sim_before,
        os.path.join(args.out_results, "similarity_matrix_before_training.png"),
        "Cosine Similarity — Random Encoder (Before Training)", n=16
    )

    # ── Train SimCLR ──────────────────────────────────────────────────────
    print(f"\nStarting SimCLR pretraining for {args.epochs} epochs...\n")
    train_simclr(model, loader, optimizer, criterion, device,
                 args.epochs, args.out_graphs, args.out_results, args.models_dir)

    # ── Similarity AFTER training ─────────────────────────────────────────
    print("\nComputing similarity AFTER training...")
    z1_after, z2_after = get_features(model.encoder, v1_sample, v2_sample, device)
    same_after, diff_after = compute_avg_similarities(z1_after, z2_after)

    sim_after = torch.mm(
        torch.cat([z1_after, z2_after], dim=0),
        torch.cat([z1_after, z2_after], dim=0).t()
    ).cpu().numpy()
    plot_similarity_matrix(
        sim_after,
        os.path.join(args.out_results, "similarity_matrix_after_training.png"),
        "Cosine Similarity — SimCLR Encoder (After Training)", n=16
    )

    # ── Print comparison table ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Feature Similarity Comparison (for your report)")
    print("="*60)
    print(f"  {'Pair Type':<35} {'Before':>8} {'After':>8}")
    print("  " + "-"*55)
    print(f"  {'Same image, two augmented views':<35} {same_before:>8.4f} {same_after:>8.4f}")
    print(f"  {'Different images':<35} {diff_before:>8.4f} {diff_after:>8.4f}")
    print("="*60)

    # Save for metrics.json
    with open(os.path.join(args.out_results, "similarity_scores.txt"), "w") as f:
        f.write(f"same_view_similarity_before={same_before:.6f}\n")
        f.write(f"different_image_similarity_before={diff_before:.6f}\n")
        f.write(f"same_view_similarity_after={same_after:.6f}\n")
        f.write(f"different_image_similarity_after={diff_after:.6f}\n")
    print(f"\n  Scores saved to results/similarity_scores.txt")


if __name__ == "__main__":
    main()
