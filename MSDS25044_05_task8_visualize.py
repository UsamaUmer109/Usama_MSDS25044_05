"""
MSDS25044_05_task8_visualize.py
Task 8: PCA/t-SNE Feature Visualization 
+ metrics.json + test_predictions.csv

Outputs:
  results/random_encoder_pca_or_tsne.png
  results/simclr_encoder_pca_or_tsne.png
  results/finetuned_encoder_pca_or_tsne.png
  results/metrics.json
  results/test_predictions.csv

Run:
  python MSDS25044_05_task8_visualize.py \
    --data-dir data \
    --simclr-ckpt models/simclr_encoder.pt \
    --finetuned-ckpt models/finetuned_model.pt
"""

import os, sys, argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T, torchvision.models as models
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.seed import set_seed

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
CLASSES    = ["airplane","automobile","bird","cat","deer",
              "dog","frog","horse","ship","truck"]
COLORS     = plt.cm.tab10(np.linspace(0, 1, 10))

def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

def get_subset(data_dir, split_file, train, transform):
    with open(split_file) as f:
        indices = [int(l.strip()) for l in f if l.strip()]
    full = torchvision.datasets.CIFAR10(root=data_dir, train=train,
                                         download=False, transform=transform)
    return Subset(full, indices)

def build_encoder():
    base = models.resnet18(weights=None)
    base.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    base.maxpool = nn.Identity()
    return nn.Sequential(*list(base.children())[:-1])

def build_full_model():
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model

def extract_features(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbs in tqdm(loader, desc="  Extracting", leave=False):
            h = encoder(imgs.to(device)).view(imgs.size(0), -1)
            feats.append(h.cpu()); labels.append(lbs)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

def extract_fullmodel_features(model, loader, device):
    model.eval()
    feats, labels, probs_all = [], [], []
    with torch.no_grad():
        for imgs, lbs in tqdm(loader, desc="  Extracting", leave=False):
            imgs = imgs.to(device)
            feat_out = []
            def hook(m, i, o): feat_out.append(o.view(o.size(0), -1))
            h = model.avgpool.register_forward_hook(hook)
            out = model(imgs); h.remove()
            probs = torch.softmax(out, dim=1)
            feats.append(feat_out[0].cpu())
            labels.append(lbs)
            probs_all.append(probs.cpu())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy(), torch.cat(probs_all).numpy()

def plot_2d(feats, labels, out_path, title, method, seed):
    print(f"  Running {method.upper()}...")
    if method == "tsne":
        coords = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(feats)
    else:
        coords = PCA(n_components=2, random_state=seed).fit_transform(feats)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(CLASSES):
        mask = labels == i
        ax.scatter(coords[mask,0], coords[mask,1], c=[COLORS[i]],
                   label=cls, alpha=0.6, s=15)
    ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

def read_val(path, default=0.0):
    try:
        with open(path) as f: return float(f.read().strip())
    except: return default

def read_sim(path):
    d = {}
    try:
        with open(path) as f:
            for line in f:
                k, v = line.strip().split("="); d[k] = float(v)
    except: pass
    return d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",       default="data")
    parser.add_argument("--splits-dir",     default="splits")
    parser.add_argument("--simclr-ckpt",    default="models/simclr_encoder.pt")
    parser.add_argument("--finetuned-ckpt", default="models/finetuned_model.pt")
    parser.add_argument("--out-results",    default="results")
    parser.add_argument("--method",         default="tsne", choices=["tsne","pca"])
    parser.add_argument("--n-samples",      type=int, default=1000)
    parser.add_argument("--batch-size",     type=int, default=64)
    parser.add_argument("--seed",           type=int, default=2026)
    parser.add_argument("--device",         default="auto")
    parser.add_argument("--student-name",   default="Usama Umer")
    parser.add_argument("--roll-number",    default="MSDS25044")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_results, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"\nDevice: {device} | Method: {args.method.upper()} | Samples: {args.n_samples}\n")

    # Load 1000 val images with fixed seed
    val_set = get_subset(args.data_dir, f"{args.splits_dir}/val.txt", True, eval_transform())
    rng     = np.random.RandomState(args.seed)
    idx     = rng.choice(len(val_set), size=min(args.n_samples, len(val_set)), replace=False)
    loader  = DataLoader(Subset(val_set, idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Test loader for predictions
    te_set = get_subset(args.data_dir, f"{args.splits_dir}/test.txt", False, eval_transform())
    te_ld  = DataLoader(te_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 1. Random encoder
    print("="*50)
    print("1. Random Encoder Visualization")
    print("="*50)
    enc_r = build_encoder().to(device)
    f_r, l_r = extract_features(enc_r, loader, device)
    plot_2d(f_r, l_r, f"{args.out_results}/random_encoder_pca_or_tsne.png",
            f"Random Encoder ({args.method.upper()})", args.method, args.seed)

    # 2. SimCLR encoder
    print("="*50)
    print("2. SimCLR Encoder Visualization")
    print("="*50)
    enc_s = build_encoder().to(device)
    enc_s.load_state_dict(torch.load(args.simclr_ckpt, map_location=device))
    f_s, l_s = extract_features(enc_s, loader, device)
    plot_2d(f_s, l_s, f"{args.out_results}/simclr_encoder_pca_or_tsne.png",
            f"SimCLR Encoder ({args.method.upper()})", args.method, args.seed)

    # 3. Fine-tuned encoder
    print("="*50)
    print("3. Fine-tuned Encoder Visualization")
    print("="*50)
    model_ft = build_full_model().to(device)
    model_ft.load_state_dict(torch.load(args.finetuned_ckpt, map_location=device))
    f_ft, l_ft, _ = extract_fullmodel_features(model_ft, loader, device)
    plot_2d(f_ft, l_ft, f"{args.out_results}/finetuned_encoder_pca_or_tsne.png",
            f"Fine-tuned Encoder ({args.method.upper()})", args.method, args.seed)

    # test_predictions.csv
    print("\nGenerating test_predictions.csv...")
    model_ft.eval()
    preds_all, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for imgs, lbs in tqdm(te_ld, desc="  Predicting", leave=False):
            out = model_ft(imgs.to(device))
            probs_all.extend(torch.softmax(out,1).cpu().numpy())
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(lbs.numpy())

    with open(f"{args.out_results}/test_predictions.csv", "w") as f:
        f.write("image_index,true_label,predicted_label," +
                ",".join([f"prob_class_{i}" for i in range(10)]) + "\n")
        for i, (t, p, pr) in enumerate(zip(labels_all, preds_all, probs_all)):
            f.write(f"{i},{t},{p}," + ",".join([f"{x:.6f}" for x in pr]) + "\n")
    print(f"  Saved: {args.out_results}/test_predictions.csv")

    # metrics.json
    sim = read_sim(f"{args.out_results}/similarity_scores.txt")
    lp  = read_sim(f"{args.out_results}/linear_probe_accs.txt")
    metrics = {
        "student_name":                      args.student_name,
        "roll_number":                       args.roll_number,
        "seed":                              2026,
        "batch_size":                        64,
        "simclr_epochs":                     50,
        "linear_probe_epochs":               20,
        "finetuning_epochs":                 20,
        "learning_rate":                     0.0003,
        "temperature":                       0.5,
        "supervised_10percent_test_acc":     round(read_val(f"{args.out_results}/supervised_test_acc.txt"), 4),
        "random_linear_probe_test_acc":      round(lp.get("random_linear_probe_test_acc", 0.0), 4),
        "simclr_linear_probe_test_acc":      round(lp.get("simclr_linear_probe_test_acc", 0.0), 4),
        "simclr_finetune_test_acc":          round(read_val(f"{args.out_results}/finetune_test_acc.txt"), 4),
        "same_view_similarity_before":       round(sim.get("same_view_similarity_before", 0.0), 4),
        "different_image_similarity_before": round(sim.get("different_image_similarity_before", 0.0), 4),
        "same_view_similarity_after":        round(sim.get("same_view_similarity_after", 0.0), 4),
        "different_image_similarity_after":  round(sim.get("different_image_similarity_after", 0.0), 4),
    }
    with open(f"{args.out_results}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {args.out_results}/metrics.json")

    print("\n" + "="*65)
    print("  FINAL RESULTS TABLE")
    print("="*65)
    print(f"  {'Model':<40} {'Test Acc':>10}")
    print("  " + "-"*52)
    print(f"  {'Supervised ResNet-18 (10% labels)':<40} {metrics['supervised_10percent_test_acc']*100:>9.2f}%")
    print(f"  {'Random encoder + linear probe':<40} {metrics['random_linear_probe_test_acc']*100:>9.2f}%")
    print(f"  {'SimCLR encoder + linear probe':<40} {metrics['simclr_linear_probe_test_acc']*100:>9.2f}%")
    print(f"  {'SimCLR encoder + fine-tuning':<40} {metrics['simclr_finetune_test_acc']*100:>9.2f}%")
    print("="*65)

if __name__ == "__main__":
    main()
