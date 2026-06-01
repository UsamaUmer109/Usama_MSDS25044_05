"""
MSDS25044_05_task7_finetune.py
Task 7: Fine-tuning the SimCLR Encoder

Initializes encoder with SimCLR weights, trains full model end-to-end
using only 10% labeled data.

Outputs:
  graphs/finetuning_accuracy.png
  models/finetuned_model.pt

Run:
  python MSDS25044_05_task7_finetune.py \
    --data-dir data \
    --simclr-ckpt models/simclr_encoder.pt
"""

import os, sys, argparse
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T, torchvision.models as models
from torch.utils.data import DataLoader, Subset
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.seed import set_seed

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def train_transform():
    return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

def get_subset(data_dir, split_file, train, transform):
    with open(split_file) as f:
        indices = [int(l.strip()) for l in f if l.strip()]
    full = torchvision.datasets.CIFAR10(root=data_dir, train=train,
                                         download=False, transform=transform)
    return Subset(full, indices)

def build_full_model():
    """ResNet-18 modified for CIFAR-10 with classification head."""
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    c, t, total_loss = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, desc="  Train" if train else "  Eval ", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            if train: optimizer.zero_grad()
            out = model(imgs); loss = criterion(out, labels)
            if train: loss.backward(); optimizer.step()
            total_loss += loss.item()*imgs.size(0)
            c += (out.argmax(1)==labels).sum().item(); t += labels.size(0)
    return total_loss/t, c/t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="data")
    parser.add_argument("--splits-dir",  default="splits")
    parser.add_argument("--simclr-ckpt", default="models/simclr_encoder.pt")
    parser.add_argument("--out-results", default="results")
    parser.add_argument("--out-graphs",  default="graphs")
    parser.add_argument("--models-dir",  default="models")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--seed",        type=int,   default=2026)
    parser.add_argument("--device",      default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_results, exist_ok=True)
    os.makedirs(args.out_graphs,  exist_ok=True)
    os.makedirs(args.models_dir,  exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"\nDevice: {device}\n")

    tr_ld = DataLoader(get_subset(args.data_dir, f"{args.splits_dir}/train_labeled_10percent.txt", True,  train_transform()), batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    vl_ld = DataLoader(get_subset(args.data_dir, f"{args.splits_dir}/val.txt",   True,  eval_transform()), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    te_ld = DataLoader(get_subset(args.data_dir, f"{args.splits_dir}/test.txt",  False, eval_transform()), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Load SimCLR encoder weights
    model    = build_full_model().to(device)
    sim_w    = torch.load(args.simclr_ckpt, map_location=device)
    model_d  = model.state_dict()
    matched  = {k: v for k, v in sim_w.items() if k in model_d and model_d[k].shape == v.shape}
    model_d.update(matched); model.load_state_dict(model_d)
    print(f"  Loaded {len(matched)}/{len(model_d)} layers from SimCLR checkpoint\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    tr_accs, vl_accs, best_val = [], [], 0.0
    best_ckpt = f"{args.models_dir}/finetuned_model.pt"

    print(f"Fine-tuning for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = run_epoch(model, tr_ld, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, vl_ld, criterion, optimizer, device, train=False)
        tr_accs.append(tr_acc); vl_accs.append(vl_acc)
        print(f"  Epoch {epoch:2d}/{args.epochs} | "
              f"Train loss={tr_loss:.4f} acc={tr_acc*100:.1f}% | "
              f"Val   loss={vl_loss:.4f} acc={vl_acc*100:.1f}%")
        if vl_acc > best_val:
            best_val = vl_acc
            torch.save(model.state_dict(), best_ckpt)

    # Test
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, test_acc = run_epoch(model, te_ld, criterion, optimizer, device, train=False)
    print(f"\n  Fine-tuned Test Accuracy: {test_acc*100:.2f}%")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs+1), [x*100 for x in tr_accs], label="Train Acc", color="steelblue", linewidth=2)
    plt.plot(range(1, args.epochs+1), [x*100 for x in vl_accs], label="Val Acc",   color="tomato",    linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title("Fine-tuning — SimCLR Pretrained Encoder")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{args.out_graphs}/finetuning_accuracy.png", dpi=150)
    plt.close()
    print(f"  Saved: {args.out_graphs}/finetuning_accuracy.png")

    with open(f"{args.out_results}/finetune_test_acc.txt", "w") as f:
        f.write(str(round(test_acc, 6)))

    print("\n  Upload models/finetuned_model.pt to Google Drive.")

if __name__ == "__main__":
    main()
