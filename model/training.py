import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torchvision import transforms
from model.emotion_net import EmotionCNN


class EmotionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=0)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The first column is the label (emotion), the rest are pixel values
        label = int(self.data.iloc[idx, 0])
        pixel_string = self.data.iloc[idx, 1]
        pixels = np.array([float(p) for p in pixel_string.split()]).reshape(48, 48).astype(np.float32)

        # Convert to tensor and apply any additional transforms
        if self.transform:
            pixels = self.transform(pixels)

        return pixels, label


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass into the network
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


if __name__ == '__main__':

    # Path parameters
    train_csv = "../datasets/train_aug.csv"
    test_csv = "../datasets/private_test.csv"
    model_save_path = "../cnn.pth"

    # Hyperparameters
    batch_size = 64
    LR = 0.001
    n_epochs = 25

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transformation method to make images tensors
    transform = transforms.ToTensor()

    # Load datasets from CSV
    train_dataset = EmotionDataset(csv_file=train_csv, transform=transform)
    test_dataset = EmotionDataset(csv_file=test_csv, transform=transform)

    # Dataloaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    """
    Initialization of our defined model, the loss function (cross entropy), and optimizer - stochastic gradient
    descent (SGD)
    """
    model = EmotionCNN(num_of_channels=1, num_of_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    # Training loop

    import time

    start_time = time.time()

    best_acc = 0.0
    for epoch in range(3):
        print(f'Epoch {epoch + 1}/{n_epochs}')

        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        # Save the best model based on test accuracy
        if test_acc > best_acc:
            torch.save(model.state_dict(), model_save_path)
            best_acc = test_acc

    end_time = time.time()
    print(f'Highest test accuracy: {best_acc}')
    print(f'Time spent training: '
          f'{int((end_time - start_time) // 3600)}:{int((end_time % start_time) // 60)}:{int(end_time % 60)}')
