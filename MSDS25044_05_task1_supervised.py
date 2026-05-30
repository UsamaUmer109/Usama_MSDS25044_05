"""
MSDS25044_05_task1_supervised.py
----------------------------------
Task 1: Supervised Baseline with Limited Labels

Trains ResNet-18 from scratch on the 10% labeled CIFAR-10 split.

Outputs:
  graphs/supervised_loss.png
  results/supervised_confusion_matrix.png

Quick test (1 epoch):
  python MSDS25044_05_task1_supervised.py --data-dir data/cifar-10-batches-py --epochs 1

Full run (50 epochs):
  python MSDS25044_05_task1_supervised.py --data-dir data/cifar-10-batches-py --epochs 50
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# ── import dataset utility ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset_splits import get_cifar10_subset
from utils.seed import set_seed

# ── constants ─────────────────────────────────────────────────────────────────
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
CLASSES    = ["airplane","automobile","bird","cat","deer",
              "dog","frog","horse","ship","truck"]


# ── ResNet-18 modified for CIFAR-10 (32×32) ───────────────────────────────────
def build_resnet18_cifar(num_classes=10):
    """
    Required architecture from assignment Section 5:
      - conv1: 3×3, stride 1, padding 1
      - remove maxpool
      - fc: 512 → num_classes
    No pretrained weights allowed.
    """
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, num_classes)
    return model


# ── transforms ────────────────────────────────────────────────────────────────
def train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])

def eval_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


# ── one training epoch ────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Eval ", leave=False):
            images, labels = images.to(device), labels.to(device)
            out  = model(images)
            loss = criterion(out, labels)
            total_loss += loss.item() * images.size(0)
            preds       = out.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ── plots ─────────────────────────────────────────────────────────────────────
def save_loss_plot(train_losses, val_losses, path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="steelblue", linewidth=2)
    plt.plot(epochs, val_losses,   label="Val Loss",   color="tomato",    linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Supervised Baseline — Train & Val Loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

def save_confusion_matrix(labels, preds, path):
    cm   = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, colorbar=True, cmap="Blues")
    plt.title("Supervised Baseline — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="data/cifar-10-batches-py")
    parser.add_argument("--splits-dir",  default="splits")
    parser.add_argument("--out-results", default="results")
    parser.add_argument("--out-graphs",  default="graphs")
    parser.add_argument("--models-dir",  default="models")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--seed",        type=int,   default=2026)
    parser.add_argument("--device",      default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_results, exist_ok=True)
    os.makedirs(args.out_graphs,  exist_ok=True)
    os.makedirs(args.models_dir,  exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"\nDevice : {device}")
    print(f"Epochs : {args.epochs}  |  Batch : {args.batch_size}  |  LR : {args.lr}\n")

    # ── loaders get_cifar10_subset ──────────────────────────────
    train_set = get_cifar10_subset(
        data_root=args.data_dir,
        split_file=os.path.join(args.splits_dir, "train_labeled_10percent.txt"),
        train=True,
        transform=train_transform(),
    )
    val_set = get_cifar10_subset(
        data_root=args.data_dir,
        split_file=os.path.join(args.splits_dir, "val.txt"),
        train=True,          # val indices are inside the official train split
        transform=eval_transform(),
    )
    test_set = get_cifar10_subset(
        data_root=args.data_dir,
        split_file=os.path.join(args.splits_dir, "test.txt"),
        train=False,         # test indices are inside the official test split
        transform=eval_transform(),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train : {len(train_set):,} images")
    print(f"Val   : {len(val_set):,} images")
    print(f"Test  : {len(test_set):,} images\n")

    # ── model / loss / optimiser ───────────────────────────────────────────
    model     = build_resnet18_cifar().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── training loop ──────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc          = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _   = evaluate(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train  loss={tr_loss:.4f}  acc={tr_acc*100:.1f}% | "
              f"Val  loss={vl_loss:.4f}  acc={vl_acc*100:.1f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            ckpt = os.path.join(args.models_dir, "supervised_best.pt")
            torch.save(model.state_dict(), ckpt)

    # ── test evaluation ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(
        os.path.join(args.models_dir, "supervised_best.pt"), map_location=device))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*55}")
    print(f"  Final Test Accuracy : {test_acc*100:.2f}%")
    print(f"{'='*55}")

    # ── save outputs ───────────────────────────────────────────────────────
    save_loss_plot(train_losses, val_losses,
                   os.path.join(args.out_graphs, "supervised_loss.png"))
    save_confusion_matrix(labels, preds,
                          os.path.join(args.out_results, "supervised_confusion_matrix.png"))

    # save accuracy for metrics.json later
    with open(os.path.join(args.out_results, "supervised_test_acc.txt"), "w") as f:
        f.write(str(round(test_acc, 6)))

    print(f"\n Upload models/supervised_best.pt to Google Drive.")


if __name__ == "__main__":
    main()
