"""Train a sign language classifier from a landmark CSV.

CSV format (no header):
  label, source, frame_id, feat_0, feat_1, ..., feat_62

Example:
  A, kaggle, img001, 0.012, -0.034, ...

Usage:
  python training/train_classifier.py \
    --dataset datasets/asl/landmarks.csv \
    --epochs 30 \
    --lr 3e-4 \
    --batch-size 64 \
    --checkpoint-dir checkpoints/asl
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

from training.model_one_hand import OneHandClassifier
from training.augment import augment_one_hand


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

LETTERS = [chr(i) for i in range(65, 91)]  # A-Z
LABEL_TO_IDX = {l: i for i, l in enumerate(LETTERS)}


class LandmarkDataset(Dataset):
    """Read pre-extracted landmark CSVs."""

    def __init__(self, csv_path: str, transform=None):
        import pandas as pd

        df = pd.read_csv(csv_path, header=None, dtype={0: str, 1: str, 2: str})
        labels = df.iloc[:, 0].apply(lambda l: LABEL_TO_IDX[l.upper()]).values
        n_meta = 3
        features = df.iloc[:, n_meta:].astype(np.float32).values
        self.labels = torch.from_numpy(labels).long()
        self.features = torch.from_numpy(features)
        self.transform = transform
        self.n_features = self.features.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.features[idx].clone()
        label = self.labels[idx]
        if self.transform:
            feat = self.transform(feat.numpy(), self.hands)
        return feat, label


# ---------------------------------------------------------------------------
# Augmentation pipeline (called from DataLoader)
# ---------------------------------------------------------------------------

def make_augmentation(transform_name: str = "landmark_noise"):
    """Return a callable(x: np.ndarray) -> np.ndarray."""
    rng = np.random.default_rng()

    def apply(x: np.ndarray):
        if transform_name == "none":
            return x.astype(np.float32)
        return augment_one_hand(x, rng)

    return apply


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 26):
    return OneHandClassifier(num_classes)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train sign-language classifier")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to landmark CSV")
    parser.add_argument("--num-classes", type=int, default=26)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Where to save model checkpoints")
    parser.add_argument("--augment", choices=["landmark_noise", "none"],
                        default="landmark_noise",
                        help="Augmentation to use on training data")
    parser.add_argument("--csv-file", type=str, default="train_log.csv",
                        help="CSV file for epoch-level logging")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Dataset + split
    full_ds = LandmarkDataset(args.dataset)
    print(f"Loaded {len(full_ds)} samples, "
          f"{full_ds.n_features} features")

    n_total = len(full_ds)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Apply augmentation to training split
    train_ds.dataset.transform = make_augmentation(args.augment)
    val_ds.dataset.transform = make_augmentation("none")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4,
                            pin_memory=True)

    # Model + optimiser
    model = build_model(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"LR: {args.lr}, WD: {args.weight_decay}, Augment: {args.augment}")
    print()

    # Training loop
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.csv_file)

    # Initialise log CSV
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion,
                                    optimizer, device)
        v_loss, v_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{t_loss:.6f},{t_acc:.6f},{v_loss:.6f},{v_acc:.6f}\n")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            ckpt = os.path.join(args.checkpoint_dir, "best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  ** saved best checkpoint (val_acc={v_acc:.4f})")

    print(f"\nDone. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {args.checkpoint_dir}/best.pt")
    print(f"Log:        {log_path}")


if __name__ == "__main__":
    main()
