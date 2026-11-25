"""Fine-tune DermClassifier on train/ and val/ ImageFolder splits."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import SKIN_CLASSES, DermClassifier, get_preprocessing_transform


class DermImageFolder(ImageFolder):
    """Use the model's label order rather than ImageFolder's alphabetical order."""

    def find_classes(self, directory):
        classes, _ = super().find_classes(directory)
        if set(classes) != set(SKIN_CLASSES):
            raise ValueError(f"{directory} must contain exactly these class folders: {SKIN_CLASSES}")
        return list(SKIN_CLASSES), {name: index for index, name in enumerate(SKIN_CLASSES)}


def make_dataset(path):
    return DermImageFolder(path, transform=get_preprocessing_transform())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    num_classes = len(SKIN_CLASSES)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for images, labels in loader:
        logits, _ = model(images.to(device))
        predicted = logits.argmax(1).cpu()
        confusion += torch.bincount(
            labels * num_classes + predicted, minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
    correct = confusion.diag()
    support = confusion.sum(dim=1)
    recall = correct / support
    return {
        "validation_accuracy": (correct.sum() / support.sum()).item(),
        "balanced_accuracy": recall.mean().item(),
        "per_class_recall": dict(zip(SKIN_CLASSES, recall.tolist())),
    }


def train(data_dir, output, *, epochs=5, batch_size=16, lr=1e-4, pretrained=True,
          device="cpu", seed=42):
    """Train the classifier, report held-out split metrics, and save its state dict."""
    if epochs < 1 or batch_size < 2 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Use at least one epoch, batch size >= 2, and a finite positive learning rate")
    torch.manual_seed(seed)
    data_dir = Path(data_dir)
    train_data = make_dataset(data_dir / "train")
    val_data = make_dataset(data_dir / "val")
    # BatchNorm in the classifier requires at least two training examples per batch.
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        drop_last=len(train_data) % batch_size == 1,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size)
    model = DermClassifier(num_classes=len(SKIN_CLASSES), pretrained=pretrained).to(device)
    # Only the classification head is supervised; never report the auxiliary head.
    model.confidence_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        print(json.dumps({"epoch": epoch, **metrics}), flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Start without ImageNet weights")
    args = parser.parse_args()
    train(args.data_dir, args.output, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, pretrained=not args.no_pretrained, device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
