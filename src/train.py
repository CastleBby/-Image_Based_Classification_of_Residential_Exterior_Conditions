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
- Printed training loss per epoch
- Trained model (in memory; optional saving can be added)
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import get_model


def train(config="baseline", epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, classes = get_dataloaders(config=config)

    model = get_model(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train(config="baseline")