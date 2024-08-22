import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_optimizer_and_scheduler(model, learning_rate=0.001, num_epochs=10, warmup_epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    return optimizer, scheduler

def warmup_scheduler(optimizer, epoch, warmup_epochs=5, lr=0.001):
    if epoch < warmup_epochs:
        lr = lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs=10, warmup_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        warmup_scheduler(optimizer, epoch, warmup_epochs)
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        if epoch >= warmup_epochs:
            scheduler.step()
