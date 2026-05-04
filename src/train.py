"""
train.py

Purpose:
--------
Implements the training pipeline for the residential exterior condition
classification model. This script trains the model under different configurations
to support comparative analysis.

Inputs:
-------
- config (str): Training configuration type
- epochs (int): Number of training epochs

Outputs:
--------
- Printed training and validation metrics per epoch
- Saved trained model weights
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import numpy as np

from src.dataset import get_dataloaders
from src.model import get_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_total += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = loss_total / len(loader)

    return avg_loss, accuracy


def train(config="baseline", epochs=10):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, classes = get_dataloaders(config=config)

    model = get_model(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # tracking metrics for plotting later
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0

    print(f"\nRunning config: {config}\n")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # average training loss per epoch
        train_loss /= len(train_loader)

        # evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, device)

        # SAVE BEST MODEL ONLY

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/models/{config}_model.pth")

        # store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("-" * 40)

    # save model AFTER training completes
    os.makedirs("outputs/models", exist_ok=True)
    torch.save(model.state_dict(), f"outputs/models/{config}_model.pth")

    print(f"\nBest model saved to outputs/models/{config}_model.pth")

    return train_losses, val_losses, val_accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        choices=["baseline", "augment", "synthetic", "combined"],
        help="Training configuration"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    args = parser.parse_args()

    train(config=args.config, epochs=args.epochs)