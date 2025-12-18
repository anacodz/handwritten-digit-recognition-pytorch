import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


class MNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(MNISTCNN, self).__init__()
        # Input: (batch, 1, 28, 28)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> (32, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # -> (32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (64, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # -> (64, 7, 7)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_dataloaders(batch_size: int = 64, val_split: float = 0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST normalization
    ])

    train_full = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, dataloader, criterion, device, desc="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 64
    num_epochs = 5
    lr = 1e-3
    val_split = 0.1

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        val_split=val_split
    )

    model = MNISTCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device, desc="Val")

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("checkpoints", "best_mnist_cnn.pt"))
            print(f"Saved new best model with val acc = {best_val_acc:.4f}")

    # Final test evaluation
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device, desc="Test")
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
