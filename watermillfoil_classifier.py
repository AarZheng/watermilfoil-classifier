#!/usr/bin/env python3
"""
Watermillfoil Image Classifier — end‑to‑end PyTorch script

This single file supports:
  • Training (transfer learning)
  • Evaluation on the held‑out test set
  • Single‑image or folder inference

Works with a folder structure like ImageFolder expects:

DATA_ROOT/
  train/
    millfoil/
      img_001.jpg
      ...
    other/
      img_a.jpg
      ...
  val/
    millfoil/
    other/
  test/
    millfoil/
    other/

Example commands:
  python watermillfoil_classifier.py train \
    --data dataset --epochs 20 --model efficientnet_b0 \
    --batch-size 32 --img-size 256 --use-sampler --amp

  python watermillfoil_classifier.py evaluate \
    --data dataset --checkpoint runs/best.pt

  python watermillfoil_classifier.py predict \
    --checkpoint runs/best.pt --image some_photo.jpg

Optional: install scikit-learn to print ROC-AUC & a confusion matrix.
"""

from __future__ import annotations
import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

# Handle truncated images globally
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional metrics (if installed)
try:
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # faster on GPU for varying sizes


@dataclass
class Config:
    data_root: str
    model_name: str = "efficientnet_b0"  # or "resnet50"
    img_size: int = 256
    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = False
    freeze_backbone: bool = False
    use_sampler: bool = False  # WeightedRandomSampler for class imbalance
    class_weighted_loss: bool = False  # alternative to sampler
    patience: int = 5
    num_workers: int = 4
    out_dir: str = "runs"
    checkpoint: str | None = None


# ----------------------------
# Data & Transforms
# ----------------------------

def build_transforms(img_size: int, weights) -> Tuple[T.Compose, T.Compose]:
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std = weights.meta.get("std", [0.229, 0.224, 0.225])

    train_tfms = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),  # aquatic plants can appear inverted in photos
        T.RandomRotation(degrees=15, expand=False),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        T.RandomPerspective(distortion_scale=0.2, p=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    eval_tfms = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_tfms, eval_tfms


def make_datasets(cfg: Config, eval_only: bool = False):
    weights = EfficientNet_B0_Weights.DEFAULT if cfg.model_name.startswith("efficientnet") else ResNet50_Weights.DEFAULT
    train_tfms, eval_tfms = build_transforms(cfg.img_size, weights)

    paths = {
        'train': os.path.join(cfg.data_root, 'train'),
        'val': os.path.join(cfg.data_root, 'val'),
        'test': os.path.join(cfg.data_root, 'test'),
    }
    for split, p in paths.items():
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Expected folder not found: {p}")

    ds_train = None if eval_only else ImageFolder(paths['train'], transform=train_tfms)
    ds_val = ImageFolder(paths['val'], transform=eval_tfms)
    ds_test = ImageFolder(paths['test'], transform=eval_tfms)

    class_to_idx = ds_val.class_to_idx  # consistent mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return ds_train, ds_val, ds_test, class_to_idx, idx_to_class


def compute_sample_weights(dataset: ImageFolder) -> Tuple[List[float], List[int]]:
    # Compute inverse-frequency weights per sample for imbalanced datasets
    labels = [y for (_, y) in dataset.samples]
    num_classes = max(labels) + 1
    counts = [0] * num_classes
    for y in labels:
        counts[y] += 1
    total = len(labels)
    class_weights = [total / (c if c > 0 else 1) for c in counts]
    sample_weights = [class_weights[y] for y in labels]
    return sample_weights, counts


# ----------------------------
# Model
# ----------------------------

def build_model(cfg: Config, num_classes: int) -> Tuple[nn.Module, str]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.model_name == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        # Replace classifier head
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        weights_name = 'efficientnet_b0'
    elif cfg.model_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        weights_name = 'resnet50'
    else:
        raise ValueError(f"Unsupported model: {cfg.model_name}")

    if cfg.freeze_backbone:
        for name, p in model.named_parameters():
            if 'classifier' in name or name.startswith('fc'):
                p.requires_grad = True
            else:
                p.requires_grad = False

    model.to(device)
    return model, weights_name


# ----------------------------
# Training/Eval Utilities
# ----------------------------

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def precision_recall_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    preds = logits.argmax(dim=1)
    f1_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    eps = 1e-9
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_sum += f1
        prec_sum += precision
        rec_sum += recall
    return prec_sum / num_classes, rec_sum / num_classes, f1_sum / num_classes


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None, amp: bool = False, train: bool = True) -> Dict[str, float]:
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_prec = 0.0
    total_rec = 0.0
    total_f1 = 0.0
    n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, targets)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        acc = accuracy_from_logits(logits, targets)
        prec, rec, f1 = precision_recall_f1(logits, targets, num_classes=logits.size(1))

        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        total_prec += prec * batch_size
        total_rec += rec * batch_size
        total_f1 += f1 * batch_size
        n += batch_size
        
        # Log progress after each batch
        current_batch = n // batch_size
        total_batches = len(loader)
        print(f"  Batch {current_batch}/{total_batches} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        with open('training_log.txt', 'w' if current_batch == 1 else 'a') as f:
            f.write(f"  Batch {current_batch}/{total_batches} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | F1: {f1:.4f}\n")

    return {
        'loss': total_loss / n,
        'acc': total_acc / n,
        'precision': total_prec / n,
        'recall': total_rec / n,
        'f1': total_f1 / n,
    }


def save_checkpoint(path: str, model: nn.Module, optimizer, epoch: int, best: Dict[str, float], meta: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch,
        'best': best,
        'meta': meta,
    }
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and ckpt.get('optimizer_state') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt


# ----------------------------
# Entry Points
# ----------------------------

def train_entry(cfg: Config):
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_train, ds_val, ds_test, class_to_idx, idx_to_class = make_datasets(cfg)

    # Sampler & loaders
    if cfg.use_sampler:
        sample_weights, counts = compute_sample_weights(ds_train)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"Using WeightedRandomSampler. Class counts: {counts}")
        train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model, weights_name = build_model(cfg, num_classes=len(class_to_idx))

    # Loss
    if cfg.class_weighted_loss:
        # inverse frequency weights
        labels = [y for (_, y) in ds_train.samples]
        num_classes = len(class_to_idx)
        counts = [0] * num_classes
        for y in labels:
            counts[y] += 1
        total = len(labels)
        class_weights = torch.tensor([total / c for c in counts], dtype=torch.float32, device=device)
        print(f"Using class-weighted loss. Counts: {counts}; Weights: {[round(w,2) for w in class_weights.tolist()]}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best = {'f1': -1.0, 'acc': -1.0}
    best_path = os.path.join(cfg.out_dir, 'best.pt')

    epochs_no_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, amp=cfg.amp, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, amp=cfg.amp, train=False)
        scheduler.step(val_metrics['loss'])

        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{cfg.epochs} | {dt:.1f}s\n"
              f"  Train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} f1={train_metrics['f1']:.4f}\n"
              f"  Val  : loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} f1={val_metrics['f1']:.4f}")

        # Early stopping on F1
        if val_metrics['f1'] > best['f1']:
            best = {'f1': val_metrics['f1'], 'acc': val_metrics['acc']}
            meta = {
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class,
                'model_name': cfg.model_name,
                'img_size': cfg.img_size,
                'weights_name': weights_name,
            }
            save_checkpoint(best_path, model, optimizer, epoch, best, meta)
            print(f"  ✔ Saved new best checkpoint to {best_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print("Early stopping: no improvement.")
                break

    print("Training finished. Best (val):", best)


def evaluate_entry(cfg: Config):
    seed_everything()
    _, _, ds_test, class_to_idx, idx_to_class = make_datasets(cfg, eval_only=True)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model, _ = build_model(cfg, num_classes=len(class_to_idx))
    if not cfg.checkpoint or not os.path.exists(cfg.checkpoint):
        raise FileNotFoundError("--checkpoint is required for evaluate")

    ckpt = load_checkpoint(cfg.checkpoint, model, optimizer=None)
    print(f"Loaded checkpoint from {cfg.checkpoint} at epoch {ckpt.get('epoch')}")

    criterion = nn.CrossEntropyLoss()
    metrics = run_epoch(model, test_loader, criterion, optimizer=None, amp=cfg.amp, train=False)
    print(f"Test: loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} f1={metrics['f1']:.4f}")

    # Optional detailed metrics
    if SKLEARN_AVAILABLE:
        device = next(model.parameters()).device
        model.eval()
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x).cpu()
                all_logits.append(logits)
                all_targets.append(y)
        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        print(classification_report(targets.numpy(), preds.numpy(), target_names=[idx_to_class[i] for i in range(len(idx_to_class))]))
        if probs.size(1) == 2:
            auc = roc_auc_score(targets.numpy(), probs[:, 1].numpy())
            print(f"ROC-AUC (positive='millfoil' assumed idx 1 if so): {auc:.4f}")
        cm = confusion_matrix(targets.numpy(), preds.numpy())
        print("Confusion matrix:\n", cm)


def predict_entry(cfg: Config, image_path: str = None, folder: str = None):
    if not cfg.checkpoint or not os.path.exists(cfg.checkpoint):
        raise FileNotFoundError("--checkpoint is required for predict")

    ckpt = torch.load(cfg.checkpoint, map_location='cpu')
    meta = ckpt.get('meta', {})
    class_to_idx = meta.get('class_to_idx')
    if class_to_idx is None:
        raise RuntimeError("Checkpoint missing class_to_idx meta")
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model_name = meta.get('model_name', cfg.model_name)
    img_size = meta.get('img_size', cfg.img_size)

    # Build eval transform that matches training normalization
    if model_name == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = ResNet50_Weights.DEFAULT
    _, eval_tfms = build_transforms(img_size, weights)

    # Rebuild model & load weights
    cfg.model_name = model_name
    model, _ = build_model(cfg, num_classes=len(class_to_idx))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    from PIL import Image
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def predict_one(img_path: str):
        img = Image.open(img_path).convert('RGB')
        x = eval_tfms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
        pred_idx = int(probs.argmax().item())
        pred_class = idx_to_class[pred_idx]
        pred_conf = float(probs[pred_idx].item())
        return pred_class, pred_conf, {idx_to_class[i]: float(probs[i].item()) for i in range(len(probs))}

    paths: List[str] = []
    if image_path:
        paths.append(image_path)
    if folder:
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                paths.append(os.path.join(folder, fname))

    if not paths:
        raise ValueError("Provide --image or --folder with images to predict")

    for p in paths:
        pred_class, conf, dist = predict_one(p)
        print(f"{p}: {pred_class} (conf {conf:.3f}) | probs={dist}")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watermillfoil Image Classifier (PyTorch)")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # common options
    def add_common(p):
        p.add_argument('--data', dest='data_root', type=str, required=True, help='Path to DATA_ROOT with train/val/test folders')
        p.add_argument('--model', dest='model_name', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'resnet50'])
        p.add_argument('--img-size', dest='img_size', type=int, default=256)
        p.add_argument('--batch-size', dest='batch_size', type=int, default=32)
        p.add_argument('--num-workers', dest='num_workers', type=int, default=4)
        p.add_argument('--amp', action='store_true', help='Use mixed precision (faster on GPUs)')

    # train
    p_train = subparsers.add_parser('train')
    add_common(p_train)
    p_train.add_argument('--epochs', type=int, default=20)
    p_train.add_argument('--lr', type=float, default=3e-4)
    p_train.add_argument('--weight-decay', type=float, default=1e-4)
    p_train.add_argument('--freeze-backbone', action='store_true')
    p_train.add_argument('--use-sampler', action='store_true', help='WeightedRandomSampler for class imbalance')
    p_train.add_argument('--class-weighted-loss', action='store_true', help='CrossEntropy with class weights')
    p_train.add_argument('--patience', type=int, default=5)
    p_train.add_argument('--out-dir', type=str, default='runs')

    # evaluate
    p_eval = subparsers.add_parser('evaluate')
    add_common(p_eval)
    p_eval.add_argument('--checkpoint', type=str, required=True)

    # predict
    p_pred = subparsers.add_parser('predict')
    p_pred.add_argument('--checkpoint', type=str, required=True)
    p_pred.add_argument('--image', type=str, default=None, help='Path to a single image')
    p_pred.add_argument('--folder', type=str, default=None, help='Path to a folder of images')

    args = parser.parse_args()
    return args

def validate_dataset(data_root: str) -> bool:
    """Validate that all images in the dataset can be read properly."""
    from PIL import Image
    import os

    if not data_root or not os.path.exists(data_root):
        print(f"Error: data_root '{data_root}' does not exist")
        return False

    splits = ['train', 'val', 'test']
    classes = ['millfoil', 'other']
    invalid_files = []
    
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            print(f"Error: {split} directory not found at {split_dir}")
            return False
            
        for cls in classes:
            class_dir = os.path.join(split_dir, cls)
            if not os.path.exists(class_dir):
                print(f"Error: {cls} class directory not found at {class_dir}")
                return False
                
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify image integrity
                except Exception as e:
                    invalid_files.append((img_path, str(e)))
                    
    if invalid_files:
        print("\nFound invalid images:")
        for path, error in invalid_files:
            print(f"  {path}: {error}")
        return False
        
    print("Dataset validation successful - all images readable")
    return True

def main():
    args = parse_args()
    
    # Validate dataset for train and evaluate commands
    if args.command in ['train', 'evaluate']:
        if not validate_dataset(args.data_root):
            print("Dataset validation failed. Aborting.")
            return
    
    if args.command == 'train':
        cfg = Config(
            data_root=args.data_root,
            model_name=args.model_name,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amp=args.amp,
            freeze_backbone=args.freeze_backbone,
            use_sampler=args.use_sampler,
            class_weighted_loss=args.class_weighted_loss,
            patience=args.patience,
            num_workers=args.num_workers,
            out_dir=args.out_dir,
        )
        train_entry(cfg)
    elif args.command == 'evaluate':
        cfg = Config(
            data_root=args.data_root,
            model_name=args.model_name,
            img_size=args.img_size,
            batch_size=args.batch_size,
            amp=args.amp,
            num_workers=args.num_workers,
            checkpoint=args.checkpoint,
        )
        evaluate_entry(cfg)
    elif args.command == 'predict':
        cfg = Config(data_root='', checkpoint=args.checkpoint)  # data_root unused here
        predict_entry(cfg, image_path=args.image, folder=args.folder)


if __name__ == '__main__':
    main()
