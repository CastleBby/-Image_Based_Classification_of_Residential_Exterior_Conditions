"""
because the data is not evenly split between the 
3 labels, the train/val/test split must be done carefully 

Design Notes:
-------------

- Splitting is stratified by class to preserve class distribution
- The script halts if inconsistencies are found
- No in-place modification of original data; images are copied

Usage:
------
Run from project root:
    python src/data_split.py

"""

import os
import shutil
import random
import pandas as pd
from collections import defaultdict

# Paths
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "house_images")
CSV_PATH = os.path.join(DATA_DIR, "house_labels.csv")

SPLITS = ["train", "val", "test"]
SPLIT_RATIO = (0.7, 0.15, 0.15)


def load_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"labels.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = ["image_number", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def get_images_from_folders(image_dir):
    folder_images = defaultdict(list)

    for cls in os.listdir(image_dir):
        class_path = os.path.join(image_dir, cls)
        if not os.path.isdir(class_path):
            continue

        for img in os.listdir(class_path):
            folder_images[cls].append(img)

    return folder_images


def verify_dataset(df, folder_images):
    csv_images = set(df["image_number"])
    folder_image_set = set()

    for cls, imgs in folder_images.items():
        for img in imgs:
            folder_image_set.add(img)

    missing_in_csv = folder_image_set - csv_images
    missing_in_folders = csv_images - folder_image_set

    print("\n--- Verification Report ---")

    print(f"Total images in folders: {len(folder_image_set)}")
    print(f"Total images in CSV: {len(csv_images)}")

    if missing_in_csv:
        print(f"\nImages in folders but NOT in CSV ({len(missing_in_csv)}):")
        print(list(missing_in_csv)[:10])

    if missing_in_folders:
        print(f"\nImages in CSV but NOT in folders ({len(missing_in_folders)}):")
        print(list(missing_in_folders)[:10])

    if missing_in_csv or missing_in_folders:
        raise ValueError("Dataset mismatch detected. Fix before proceeding.")

    print("\nVerification passed: CSV matches folder contents.")


def group_by_class(df):
    class_map = defaultdict(list)

    for _, row in df.iterrows():
        class_map[row["label"]].append(row["image_number"])

    return class_map


def create_split_dirs(base_dir, classes):
    for split in SPLITS:
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)


def split_and_copy(class_map, source_dir, target_dir):
    for cls, images in class_map.items():
        random.shuffle(images)

        n = len(images)
        train_end = int(SPLIT_RATIO[0] * n)
        val_end = train_end + int(SPLIT_RATIO[1] * n)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, img_list in splits.items():
            for img_name in img_list:
                src_path = find_image_path(source_dir, cls, img_name)
                dst_path = os.path.join(target_dir, split, cls, img_name)

                shutil.copy(src_path, dst_path)

    print("\nData split complete.")


def find_image_path(source_dir, cls, img_name):
    path = os.path.join(source_dir, cls, img_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def main():
    random.seed(42)

    df = load_csv(CSV_PATH)
    folder_images = get_images_from_folders(IMAGE_DIR)

    verify_dataset(df, folder_images)

    class_map = group_by_class(df)

    print("\nClass distribution:")
    for cls, imgs in class_map.items():
        print(f"{cls}: {len(imgs)}")

    create_split_dirs(DATA_DIR, class_map.keys())

    split_and_copy(class_map, IMAGE_DIR, DATA_DIR)


if __name__ == "__main__":
    main()