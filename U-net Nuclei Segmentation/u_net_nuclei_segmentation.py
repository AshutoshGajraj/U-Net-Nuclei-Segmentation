#!/usr/bin/env python

"""PyTorch U-Net training script for the Kaggle 2018 Data Science Bowl dataset."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


SEED = 42
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = Path("stage1_train")
TEST_PATH = Path("stage1_test")
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_images_and_masks(
    train_path: Path,
    test_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    from skimage.io import imread
    from skimage.transform import resize
    from tqdm import tqdm

    train_ids = sorted(path.name for path in train_path.iterdir() if path.is_dir())
    test_ids = sorted(path.name for path in test_path.iterdir() if path.is_dir())

    x_train = np.zeros(
        (len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32
    )
    y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

    print("Resizing training images and masks")
    for index, image_id in tqdm(enumerate(train_ids), total=len(train_ids)):
        sample_path = train_path / image_id
        image = imread(sample_path / "images" / f"{image_id}.png")
        image = image[..., :IMG_CHANNELS]
        x_train[index] = resize(
            image,
            (IMG_HEIGHT, IMG_WIDTH),
            mode="constant",
            preserve_range=True,
        ) / 255.0

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        for mask_path in (sample_path / "masks").iterdir():
            mask_image = imread(mask_path)
            resized_mask = resize(
                mask_image,
                (IMG_HEIGHT, IMG_WIDTH),
                mode="constant",
                preserve_range=True,
            )
            mask = np.maximum(mask, resized_mask > 0.5)
        y_train[index] = mask

    x_test = np.zeros(
        (len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32
    )
    sizes_test: list[list[int]] = []

    print("Resizing test images")
    for index, image_id in tqdm(enumerate(test_ids), total=len(test_ids)):
        image = imread(test_path / image_id / "images" / f"{image_id}.png")
        image = image[..., :IMG_CHANNELS]
        sizes_test.append([image.shape[0], image.shape[1]])
        x_test[index] = resize(
            image,
            (IMG_HEIGHT, IMG_WIDTH),
            mode="constant",
            preserve_range=True,
        ) / 255.0

    print("Done!")
    return x_train, y_train, x_test, sizes_test


class NucleiDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: np.ndarray, masks: np.ndarray) -> None:
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)
        self.masks = torch.from_numpy(masks).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.masks[index]


class DoubleConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = DoubleConv(IMG_CHANNELS, 16, 0.1)
        self.down2 = DoubleConv(16, 32, 0.1)
        self.down3 = DoubleConv(32, 64, 0.2)
        self.down4 = DoubleConv(64, 128, 0.2)
        self.bottleneck = DoubleConv(128, 256, 0.3)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv4 = DoubleConv(256, 128, 0.2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64, 0.2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv2 = DoubleConv(64, 32, 0.1)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv1 = DoubleConv(32, 16, 0.1)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(inputs)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        c4 = self.down4(self.pool(c3))
        bottleneck = self.bottleneck(self.pool(c4))

        u4 = self.up4(bottleneck)
        u4 = self.conv4(torch.cat((u4, c4), dim=1))
        u3 = self.up3(u4)
        u3 = self.conv3(torch.cat((u3, c3), dim=1))
        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat((u2, c2), dim=1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat((u1, c1), dim=1))
        return self.output(u1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        predictions = model(images)
        loss = loss_function(predictions, masks)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def predict(
    model: nn.Module,
    images: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = torch.from_numpy(images[start : start + BATCH_SIZE]).permute(0, 3, 1, 2)
            logits = model(batch.to(device))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def main() -> None:
    import matplotlib.pyplot as plt

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train, y_train, x_test, _ = load_images_and_masks(TRAIN_PATH, TEST_PATH)
    train_dataset = NucleiDataset(x_train, y_train)
    validation_size = int(len(train_dataset) * 0.1)
    training_size = len(train_dataset) - validation_size
    train_subset, validation_subset = random_split(
        train_dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(
        validation_subset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = UNet().to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss = run_epoch(
            model, validation_loader, loss_function, device
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"train loss: {train_loss:.4f}, validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                },
                "model_for_nuclei.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 2:
                print("Early stopping")
                break

    checkpoint = torch.load("model_for_nuclei.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    predictions_train = predict(model, x_train[:training_size], device)
    predictions_validation = predict(model, x_train[training_size:], device)
    predictions_test = predict(model, x_test, device)
    predictions_train_binary = (predictions_train > 0.5).astype(np.uint8)
    predictions_validation_binary = (predictions_validation > 0.5).astype(np.uint8)
    _ = (predictions_test > 0.5).astype(np.uint8)

    index = random.randrange(training_size)
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(x_train[index])
    axes[1].imshow(y_train[index], cmap="gray")
    axes[2].imshow(predictions_train_binary[index, 0], cmap="gray")
    plt.show()

    index = random.randrange(validation_size)
    validation_index = training_size + index
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(x_train[validation_index])
    axes[1].imshow(y_train[validation_index], cmap="gray")
    axes[2].imshow(predictions_validation_binary[index, 0], cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
