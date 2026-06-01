"""
MSDS25044_05_task6_linear_probe.py
Task 6: Linear Probe Evaluation 

Experiment A: Random frozen encoder + Linear(512->10)
Experiment B: SimCLR frozen encoder + Linear(512->10)

Outputs:
  graphs/linear_probe_accuracy.png
  models/linear_probe.pt

Run:
  python MSDS25044_05_task6_linear_probe.py \
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

def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

def train_transform():
    return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

def build_encoder():
    base = models.resnet18(weights=None)
    base.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    base.maxpool = nn.Identity()
    return nn.Sequential(*list(base.children())[:-1])

def get_subset(data_dir, split_file, train, transform):
    with open(split_file) as f:
        indices = [int(l.strip()) for l in f if l.strip()]
    full = torchvision.datasets.CIFAR10(root=data_dir, train=train,
                                         download=False, transform=transform)
    return Subset(full, indices)

def extract_features(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbs in tqdm(loader, desc="  Extracting", leave=False):
            h = encoder(imgs.to(device)).view(imgs.size(0), -1)
            feats.append(h.cpu()); labels.append(lbs)
    return torch.cat(feats), torch.cat(labels)

def train_probe(tr_f, tr_l, vl_f, vl_l, device, epochs, lr, bs):
    clf = nn.Linear(512, 10).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    crt = nn.CrossEntropyLoss()
    tr_ds = torch.utils.data.TensorDataset(tr_f, tr_l)
    vl_ds = torch.utils.data.TensorDataset(vl_f, vl_l)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    vl_ld = DataLoader(vl_ds, batch_size=bs, shuffle=False)
    tr_accs, vl_accs, best_acc, best_state = [], [], 0.0, None
    for ep in range(1, epochs+1):
        clf.train(); c, t = 0, 0
        for f, l in tr_ld:
            f, l = f.to(device), l.to(device)
            opt.zero_grad(); out = clf(f); loss = crt(out, l)
            loss.backward(); opt.step()
            c += (out.argmax(1)==l).sum().item(); t += l.size(0)
        tr_acc = c/t
        clf.eval(); c, t = 0, 0
        with torch.no_grad():
            for f, l in vl_ld:
                f, l = f.to(device), l.to(device)
                out = clf(f); c += (out.argmax(1)==l).sum().item(); t += l.size(0)
        vl_acc = c/t
        tr_accs.append(tr_acc); vl_accs.append(vl_acc)
        if vl_acc > best_acc: best_acc = vl_acc; best_state = clf.state_dict()
        print(f"  Epoch {ep:2d}/{epochs} | Train {tr_acc*100:.1f}% | Val {vl_acc*100:.1f}%")
    clf.load_state_dict(best_state)
    return clf, tr_accs, vl_accs

def get_test_acc(clf, feats, labels, device):
    clf.eval()
    ds = torch.utils.data.TensorDataset(feats, labels)
    ld = DataLoader(ds, batch_size=128, shuffle=False)
    c, t = 0, 0
    with torch.no_grad():
        for f, l in ld:
            f, l = f.to(device), l.to(device)
            c += (clf(f).argmax(1)==l).sum().item(); t += l.size(0)
    return c/t

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

    tr_ld = DataLoader(get_subset(args.data_dir, f"{args.splits_dir}/train_labeled_10percent.txt", True,  train_transform()), batch_size=args.batch_size, shuffle=False, num_workers=2)
    vl_ld = DataLoader(get_subset(args.data_dir, f"{args.splits_dir}/val.txt",   True,  eval_transform()), batch_size=args.batch_size, shuffle=False, num_workers=2)
    te_ld = DataLoader(get_subset(args.data_dir, f"{args.splits_dir}/test.txt",  False, eval_transform()), batch_size=args.batch_size, shuffle=False, num_workers=2)

    results = {}

    # Experiment A — Random frozen encoder
    print("="*55)
    print("Experiment A: Random Encoder Linear Probe")
    print("="*55)
    enc_r = build_encoder().to(device)
    for p in enc_r.parameters(): p.requires_grad = False
    tr_f, tr_l = extract_features(enc_r, tr_ld, device)
    vl_f, vl_l = extract_features(enc_r, vl_ld, device)
    te_f, te_l = extract_features(enc_r, te_ld, device)
    clf_r, tr_r, vl_r = train_probe(tr_f, tr_l, vl_f, vl_l, device, args.epochs, args.lr, args.batch_size)
    acc_r = get_test_acc(clf_r, te_f, te_l, device)
    print(f"\n  Random Encoder Test Accuracy: {acc_r*100:.2f}%\n")
    results["Random Encoder"] = (tr_r, vl_r)
    torch.save(clf_r.state_dict(), f"{args.models_dir}/linear_probe_random.pt")

    # Experiment B — SimCLR frozen encoder
    print("="*55)
    print("Experiment B: SimCLR Encoder Linear Probe")
    print("="*55)
    enc_s = build_encoder().to(device)
    enc_s.load_state_dict(torch.load(args.simclr_ckpt, map_location=device))
    for p in enc_s.parameters(): p.requires_grad = False
    tr_f2, _ = extract_features(enc_s, tr_ld, device)
    vl_f2, _ = extract_features(enc_s, vl_ld, device)
    te_f2, _ = extract_features(enc_s, te_ld, device)
    clf_s, tr_s, vl_s = train_probe(tr_f2, tr_l, vl_f2, vl_l, device, args.epochs, args.lr, args.batch_size)
    acc_s = get_test_acc(clf_s, te_f2, te_l, device)
    print(f"\n  SimCLR Encoder Test Accuracy: {acc_s*100:.2f}%\n")
    results["SimCLR Encoder"] = (tr_s, vl_s)
    torch.save(clf_s.state_dict(), f"{args.models_dir}/linear_probe.pt")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["steelblue", "tomato"]
    for ax_i, split in enumerate(["Train", "Val"]):
        ax = axes[ax_i]
        for i, (lbl, (tr, vl)) in enumerate(results.items()):
            data = tr if split == "Train" else vl
            ax.plot(range(1, len(data)+1), [x*100 for x in data],
                    label=lbl, color=colors[i], linewidth=2)
        ax.set_title(f"Linear Probe — {split} Accuracy")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.out_graphs}/linear_probe_accuracy.png", dpi=150)
    plt.close()
    print(f"  Saved: {args.out_graphs}/linear_probe_accuracy.png")

    with open(f"{args.out_results}/linear_probe_accs.txt", "w") as f:
        f.write(f"random_linear_probe_test_acc={acc_r:.6f}\n")
        f.write(f"simclr_linear_probe_test_acc={acc_s:.6f}\n")

    print("\n" + "="*55)
    print(f"  Random Encoder Linear Probe : {acc_r*100:.2f}%")
    print(f"  SimCLR Encoder Linear Probe : {acc_s*100:.2f}%")
    print("="*55)
    print("\n  Upload models/linear_probe.pt to Google Drive.")

if __name__ == "__main__":
    main()
