# DL Assignment 5 — SimCLR on CIFAR-10
**Student:** Usama Umer
**Roll Number:** MSDS25044

---

## Project Structure
```
Usama_MSDS25044_05/
├── splits/
│   ├── train_ssl_unlabeled.txt
│   ├── train_labeled_10percent.txt
│   ├── val.txt
│   └── test.txt
├── data/                                   
│   └── cifar-10-batches-py/
├── models/                                 
├── results/                                ← Output images and metrics
├── graphs/                                 ← Loss/accuracy plots
├── utils/
│   ├── seed.py
│   └── dataset_splits.py
├── MSDS25044_05_task1_supervised.py        ← Task 1: Supervised baseline
├── MSDS25044_05_task2_augmentations.py     ← Task 2: Augmentation visualization
├── MSDS25044_05_task3_similarity.py        ← Task 3: Feature similarity before training
├── MSDS25044_05_task4_simclr.py            ← Task 4+5: SimCLR implementation + pretraining
├── MSDS25044_05_task6_linear_probe.py      ← Task 6: Linear probe evaluation
├── MSDS25044_05_task7_finetune.py          ← Task 7: Fine-tuning
├── MSDS25044_05_task8_visualize.py         ← Task 8: PCA/t-SNE + metrics.json
├── requirements.txt
└── Report.pdf
```

---

## Important Rules
- Do NOT add `data/` or `models/` to GitHub
- Upload all model checkpoints to Google Drive
- Make regular meaningful commits — one big commit = rejection

---

## Google Drive Model Links
- `supervised_best.pt`: https://drive.google.com/file/d/1Ee-xq2Snf5sNY4jxv8hs8u_x5fHqlUMY/view?usp=drive_link
- `simclr_encoder.pt`: https://drive.google.com/file/d/1YX0PcCIiIt4K-a04P_9-qmcUch3QYCXL/view?usp=drive_link
- `linear_probe.pt`: https://drive.google.com/file/d/1j7T00N_OlWBXUpuHMG1ukSvaGsnbZq3N/view?usp=drive_link
- `finetuned_model.pt`: https://drive.google.com/file/d/1xwKCFP2JmL5C-sy_BsAsYO7PEPRnJaFc/view?usp=sharing

---

## Setup
```bash
pip install -r requirements.txt
```

---

## How to Run Each Task

### Task 1 — Supervised Baseline
```bash
python MSDS25044_05_task1_supervised.py \
  --data-dir data --epochs 50 --batch-size 64
```
Outputs: `graphs/supervised_loss.png`, `results/supervised_confusion_matrix.png`

---

### Task 2 — Augmentation Visualization
```bash
python MSDS25044_05_task2_augmentations.py --data-dir data
```
Outputs: `results/augmentation_examples.png`

---

### Task 3 — Feature Similarity Before Training
```bash
python MSDS25044_05_task3_similarity.py --data-dir data
```
Outputs: `results/similarity_matrix_before_training.png`

---

### Task 4 + 5 — SimCLR Implementation and Pretraining
```bash
python MSDS25044_05_task4_simclr.py \
  --data-dir data --epochs 50 --batch-size 64
```
Outputs: `graphs/simclr_pretraining_loss.png`, `results/similarity_matrix_after_training.png`, `models/simclr_encoder.pt`

---

### Task 6 — Linear Probe Evaluation
```bash
python MSDS25044_05_task6_linear_probe.py \
  --data-dir data \
  --simclr-ckpt models/simclr_encoder.pt
```
Outputs: `graphs/linear_probe_accuracy.png`

---

### Task 7 — Fine-tuning
```bash
%cd /content/Usama_MSDS25044_05
!python MSDS25044_05_task7_finetune.py \
  --data-dir data \
  --simclr-ckpt models/simclr_encoder_fixed.pt
```
Outputs: `graphs/finetuning_accuracy.png`, `models/finetuned_model.pt`

---

### Task 8 — PCA/t-SNE Visualization + metrics.json
```bash
python MSDS25044_05_task8_visualize.py \
  --data-dir data \
  --simclr-ckpt models/simclr_encoder.pt \
  --finetuned-ckpt models/finetuned_model.pt \
  --method tsne
```
Outputs: `results/random_encoder_pca_or_tsne.png`, `results/simclr_encoder_pca_or_tsne.png`, `results/finetuned_encoder_pca_or_tsne.png`, `results/metrics.json`, `results/test_predictions.csv`

---

## Hyperparameter Table

| Setting | Value |
|---|---|
| Batch size | 64 |
| SimCLR epochs | 50 |
| Linear probe epochs | 20 |
| Fine-tuning epochs | 20 |
| Learning rate | 3e-4 |
| Optimizer | Adam |
| Temperature (τ) | 0.5 |
| Projection dim | 128 |
| Random seed | 2026 |
| Device | GPU (T4 on Colab) |

---

## Results Summary

| Model | Test Accuracy |
|---|---|
| Random encoder + linear probe | 23.66% |
| SimCLR encoder + linear probe | 74.87% |
| SimCLR encoder + fine-tuning | 70.06% |

---

## Commit History (Checkpoint-based)

### Checkpoint 1
- `initial project structure`
- `added CIFAR-10 split loader`
- `implemented augmentation pipeline`
- `added two-view transform`
- `generated augmentation examples`
- `implemented supervised baseline`
- `generated supervised loss plot`
- `generated supervised confusion matrix`

### Checkpoint 2
- `implemented encoder and projection head`
- `implemented cosine similarity matrix`
- `implemented NT-Xent loss`
- `added positive pair indexing`
- `generated similarity matrix before training`

### Checkpoint 3
- `implemented SimCLR training loop`
- `added contrastive pretraining`
- `generated SimCLR loss plots`
- `generated similarity matrix after training`

### Checkpoint 4
- `implemented linear probing`
- `generated linear probe accuracy plot`
- `generated linear probe results`
- `added fine-tuning pipeline`
- `generated fine-tuning results`
- `generated PCA/t-SNE visualizations`
- `added metrics.json and test_predictions.csv`
- `completed final report`
