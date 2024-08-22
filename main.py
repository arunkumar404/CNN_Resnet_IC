import torch
import torch.nn as nn
from models.resnet import resnet18
from data.dataset import get_data_loaders
from utils.training import get_optimizer_and_scheduler, train_model

def main():
    # Hyperparameters
    batch_size = 100
    learning_rate = 0.001
    num_epochs = 10
    warmup_epochs = 5
    num_classes = 10

    # Data Loaders
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Model
    model = resnet18(num_classes=num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, learning_rate, num_epochs, warmup_epochs)

    # Train
    train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs=num_epochs, warmup_epochs=warmup_epochs)

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')

if __name__ == '__main__':
    main()
