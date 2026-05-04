import os
import torch
from torchvision import datasets, transforms


def get_transforms(config="baseline"):
    """
    Returns image transformations based on configuration.
    """

    if config == "baseline":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225] # normalize image data 
            )
        ])

    elif config == "augment":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                    saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    elif config == "synthetic":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5)],
                p=0.5) # apply probabilistically
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    elif config == "combined":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5)],
                p=0.5)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:
        raise ValueError(f"Unknown config: {config}")


def get_dataloaders(data_dir="data", config="baseline", batch_size=8):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.
    """

    train_transform = get_transforms(config)
    eval_transform = get_transforms("baseline")

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset.classes