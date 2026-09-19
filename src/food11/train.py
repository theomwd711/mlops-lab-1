"""
src/food11/train.py

Trains a ResNet18 on the Food-11 dataset and logs the run to mlflow.

Usage:
    uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
"""

import argparse
import time

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

NUM_CLASSES = 11

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "food11"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet18 on Food-11")
    parser.add_argument(
        "--dataset",
        choices=["processed", "mini"],
        default="mini",
        help="Which processed dataset to train on: 'processed' (full) or 'mini' (capped, fast).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def get_dataloaders(dataset_choice: str, batch_size: int):
    root = f"./data/food11_processed_mini" if dataset_choice == "mini" else "./data/food11_processed"

    # Standard ImageNet normalization since we're using an ImageNet-pretrained backbone
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = datasets.ImageFolder(f"{root}/training", transform=transform)
    val_ds = datasets.ImageFolder(f"{root}/validation", transform=transform)
    test_ds = datasets.ImageFolder(f"{root}/evaluation", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_loader, val_loader, test_loader = get_dataloaders(args.dataset, args.batch_size)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
            }
        )

        for epoch in range(args.epochs):
            start = time.time()

            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            elapsed = time.time() - start
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_accuracy={val_acc:.4f} ({elapsed:.1f}s)"
            )

        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"Final test_accuracy={test_acc:.4f}")

        # mlflow needs a sample input to trace/save the model's graph
        sample_images, _ = next(iter(test_loader))
        input_example = sample_images[:1].cpu().numpy()

        mlflow.pytorch.log_model(model, name="model", input_example=input_example, serialization_format="pickle")


if __name__ == "__main__":
    main()
