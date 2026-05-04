"""
dataset.py

Purpose:
--------
Handles data loading and preprocessing for the residential exterior condition
classification task. This module defines transformation pipelines and constructs
PyTorch DataLoaders for training, validation, and testing.

Inputs:
-------
- data_dir (str): Root directory containing dataset splits
- config (str): Training configuration ("baseline", "augment", "synthetic", "combined")
- batch_size (int): Number of samples per batch

Outputs:
--------
- train_loader (DataLoader): Training data loader
- val_loader (DataLoader): Validation data loader
- test_loader (DataLoader): Test data loader
- classes (list): List of class names
"""

import os
import torch
from torchvision import datasets, transforms


def get_transforms(config="baseline"):
    """
    Returns image transformations for baseline and augmentation.
    """

    if config == "baseline":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    elif config == "augment":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:
        raise ValueError(f"Unknown transform config: {config}")


def get_dataloaders(data_dir="data", config="baseline", batch_size=8):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Config behavior:
    ----------------
    - baseline: original dataset, no augmentation
    - augment: original dataset with dynamic augmentation
    - synthetic: expanded dataset (pre-generated synthetic images), no augmentation
    - combined: expanded dataset with dynamic augmentation
    """

    # --- Select dataset path and transform ---
    if config == "baseline":
        train_path = os.path.join(data_dir, "train")
        train_transform = get_transforms("baseline")

    elif config == "augment":
        train_path = os.path.join(data_dir, "train")
        train_transform = get_transforms("augment")

    elif config == "synthetic":
        train_path = os.path.join(data_dir, "train_synthetic")
        train_transform = get_transforms("baseline")

    elif config == "combined":
        train_path = os.path.join(data_dir, "train_combined")
        train_transform = get_transforms("augment")

    else:
        raise ValueError(f"Invalid config: {config}")

    # --- Validation and test use baseline transforms ---
    eval_transform = get_transforms("baseline")

    # --- Datasets ---
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=eval_transform
    )

    # --- DataLoaders ---
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset.classes