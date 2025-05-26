import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn

import wandb

from models import VGG11_224

torch._functorch.config.donated_buffer = False

# Initialize Weights & Biases
EPOCHS = 250
lr = 0.0005
batch_size=128

config = {
    "learning_rate": lr,
    "architecture": "VGG11_224",
    "epochs": EPOCHS,
    "batch_size": batch_size,
    "train_dataset": "Food101",
}

wandb.init(name='deterministic', project="VGG11_224x224", config=config) # Replace with your project name
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Instantiate the model


model = VGG11_224(num_classes=101).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
model = torch.compile(model)

# Train on food, validate on flowers

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_ds = datasets.Food101(root="/home/jw7630/repos/sparsity/data", split="train", transform=trans, download=True)
test_ds = datasets.Food101(root="/home/jw7630/repos/sparsity/data", split="test", transform=trans, download=True)
ood_ds = datasets.Flowers102(root="/home/jw7630/repos/sparsity/data", split="train", transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
ood_loader = torch.utils.data.DataLoader(ood_ds, batch_size=batch_size, shuffle=False)


for epoch in range(EPOCHS):
    # TRAINING
    model.train()
    train_loss, train_accuracy = 0.0, 0.0
    for i, (images, labels) in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        y_hat = model(images)
        loss = criterion(y_hat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size
        train_accuracy += (y_hat.argmax(dim=1) == labels).float().sum().item()
    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
   
   # EVALUATION
    test_loss, test_accuracy = 0.0, 0.0
    model.eval()
    for test_images, test_labels in test_loader:
        batch_size = test_images.size(0)
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_outputs = model(test_images)

        test_loss += criterion(test_outputs, test_labels).detach().item() * batch_size
        test_accuracy += (test_outputs.argmax(dim=1) == test_labels).float().sum().cpu().item()

        del test_images, test_labels, test_outputs
        torch.cuda.empty_cache()
        optimizer.zero_grad()

    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)
    
    wandb.log({
        "train/loss": train_loss,
        "train/accuracy": train_accuracy,
        "test/loss": test_loss,
        "test/accuracy": test_accuracy,
        "epoch": epoch + 1
    })

    if (epoch+1)%50  == 0:
        torch.save(model.state_dict(), f'vggm11_{epoch+1}.pth')
torch.save(model.state_dict(), f'vggm11_{EPOCHS}.pth')

