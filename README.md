# DL Assignment 5 — SimCLR on CIFAR-10

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
├── models/                          - Saved checkpoints 
├── results/                         - Output images and metrics
├── graphs/                          - Loss/accuracy plots
├── utils/
│   ├── seed.py
│   └── dataset_splits.py
├── MSDS25044_05_task1_supervised.py
├── MSDS25044_05_task2_augmentations.py
├── MSDS25044_05_task3_similarity.py    
├── MSDS25044_05_task4_simclr.py        
├── MSDS25044_05_task5_linear_probe.py  
├── MSDS25044_05_task6_finetune.py      
├── requirements.txt
└── Report.pdf
```

## Setup
```bash
pip install -r requirements.txt
```

## IMPORTANT RULES
- Do NOT add `data/` or `models/` to GitHub
- Upload model checkpoints to Google Drive and include shareable links in this README
- Make regular, meaningful commits — one big commit = rejection

## Google Drive Links (After training)
- supervised_best.pt: https://drive.google.com/file/d/1Ee-xq2Snf5sNY4jxv8hs8u_x5fHqlUMY/view?usp=drive_link
- simclr_encoder.pt: _____________
- linear_probe.pt: _____________
- finetuned_model.pt: _____________

---

## Running the Code

### Checkpoint 1 — Dataset & Augmentations

**Step 1: Verify split loader**
```bash
python utils/dataset_splits.py data/cifar-10-batches-py splits
```

**Step 2: Run supervised baseline (quick 1-epoch test)**
```bash
python MSDS25044_05_task1_supervised.py \
  --data-dir data/cifar-10-batches-py \
  --epochs 1 --batch-size 64
```

**Step 2b: Full run (50 epochs)**
```bash
python MSDS25044_05_task1_supervised.py \
  --data-dir data/cifar-10-batches-py \
  --epochs 50 --batch-size 64
```

**Step 3: Generate augmentation examples**
```bash
python MSDS25044_05_task2_augmentations.py \
  --data-dir data/cifar-10-batches-py
```

---

### Checkpoint 2 — SimCLR Components (Day 6)
```bash
python MSDS25044_05_task4_simclr.py \
  --data-dir data/cifar-10-batches-py --mode components
```

### Checkpoint 3 — SimCLR Pretraining (Day 9)
```bash
python rollNumber_05_task4_simclr.py \
  --data-dir data/cifar-10-batches-py --epochs 50
```

### Checkpoint 4 — Linear Probe + Fine-tune (Day 12)
```bash
python MSDS25044_05_task5_linear_probe.py --data-dir data/cifar-10-batches-py
python MSDS25044_05_task6_finetune.py --data-dir data/cifar-10-batches-py
```

---

## Hyperparameter Table

| Setting | Value Used |
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

---

## Suggested Commit Schedule

### Checkpoint 1
- `initial project structure`
- `added CIFAR-10 split loader`
- `implemented supervised baseline`
- `implemented augmentation pipeline`
- `added two-view transform`
- `generated augmentation examples`

### Checkpoint 2
- `implemented ResNet-18 encoder modified for CIFAR-10`
- `implemented projection head`
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
- `added fine-tuning pipeline`
- `generated PCA/t-SNE visualizations`
- `added metrics.json and test_predictions.csv`
- `completed final report`
