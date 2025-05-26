import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import wandb

from layers import MySoftmax
from models import VGG11_224, VDP_VGG11_224
from utils_vdp import mc_nll, dirichlet_loss, gaussian_loss

torch._functorch.config.donated_buffer = False

# Initialize Weights & Biases

lr=0.0001
kl_factor = 0.000001
EPOCHS = 250
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = {
    "learning_rate": lr,
    "loss_function": "nll",
    "kl_factor": kl_factor,
    "architecture": "VGG11_224",
    "epochs": EPOCHS,
    "batch_size": batch_size,
    "train_dataset": "Food101",
    "device": device,
}
wandb.init(name="mc_nll", project="VGG11_224x224", config=config) # Replace with your project name



# Instantiate the model
model_vdp = VDP_VGG11_224(num_classes=101).to(device)
b = model_vdp(torch.randn(1, 3, 224, 224), training=True)
#model_vdp = torch.compile(model_vdp)

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



model_vdp.train()
optimizer = optim.Adam(model_vdp.parameters(), lr=lr)
model_vdp = model_vdp.to(device)
for epoch in range(EPOCHS):
    # TRAINING
    train_loss, train_kl, train_accuracy, train_average_sigma = 0.0, 0.0, 0.0, 0.0
    for i, (images, labels) in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        mu, sigma, kl = model_vdp(images)
        mu, sigma = MySoftmax()(mu, sigma)
        loss1 = mc_nll(mu, sigma, labels)
        loss = loss1 + kl_factor * kl
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size
        train_kl += kl.item() * batch_size
        train_accuracy += (mu.argmax(dim=1) == labels).float().sum().item()
        train_average_sigma += sigma.mean(1).sum().item()
    train_loss /= len(train_loader.dataset)
    train_kl /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    train_average_sigma /= len(train_loader.dataset)
   
   # EVALUATION
    test_loss, test_kl, test_accuracy, test_average_sigma = 0.0, 0.0, 0.0, 0.0
    for test_images, test_labels in test_loader:
        batch_size = test_images.size(0)
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_outputs, test_sigmas, kl = model_vdp(test_images)
        test_outputs, test_sigmas = MySoftmax()(test_outputs, test_sigmas)
        test_loss += mc_nll(test_outputs, test_sigmas, test_labels).detach().item() * batch_size
        test_kl += kl.cpu().item() * batch_size
        test_accuracy += (test_outputs.argmax(dim=1) == test_labels).float().sum().cpu().item()
        test_average_sigma += test_sigmas.mean(1).sum().cpu().item()

        del test_images, test_labels, test_outputs, test_sigmas, kl
        torch.cuda.empty_cache()
        optimizer.zero_grad()

    test_loss /= len(test_loader.dataset)
    test_kl /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)
    test_average_sigma /= len(test_loader.dataset)
    
    wandb.log({
        "train/loss": train_loss,
        "train/kl_loss": train_kl,
        "train/accuracy": train_accuracy,
        "train/average_sigma": train_average_sigma,
        "test/loss": test_loss,
        "test/kl_loss": test_kl,
        "test/accuracy": test_accuracy,
        "test/average_sigma": test_average_sigma,
        "epoch": epoch + 1
    })

    if (epoch+1)%50  == 0:
        torch.save(model_vdp.state_dict(), f'vggm11_vdp_{epoch+1}_mc_2.pth')

