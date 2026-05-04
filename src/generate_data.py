import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# --- Synthetic transforms ---
synthetic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=5)],
        p=0.5
    ),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomGrayscale(p=0.2),
])

# --- Augment transforms ---
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])


def generate_images(input_dir, output_dir, transform, copies=2):
    os.makedirs(output_dir, exist_ok=True)

    for cls in os.listdir(input_dir):
        cls_input = os.path.join(input_dir, cls)
        cls_output = os.path.join(output_dir, cls)

        os.makedirs(cls_output, exist_ok=True)

        for img_name in tqdm(os.listdir(cls_input), desc=f"{cls}"):
            if img_name.startswith("."):
                continue

            img_path = os.path.join(cls_input, img_name)
            image = Image.open(img_path).convert("RGB")

            # save original
            image.save(os.path.join(cls_output, img_name))

            # generate copies
            for i in range(copies):
                new_img = transform(image)
                new_name = f"{img_name.split('.')[0]}_gen{i}.png"
                new_img.save(os.path.join(cls_output, new_name))


if __name__ == "__main__":
    base_train = "data/train"

    # synthetic dataset
    generate_images(
        base_train,
        "data/train_synthetic",
        synthetic_transform,
        copies=2
    )

    # combined dataset
    combined_transform = transforms.Compose([
        synthetic_transform,
        augment_transform
    ])

    generate_images(
        base_train,
        "data/train_combined",
        combined_transform,
        copies=2
    )