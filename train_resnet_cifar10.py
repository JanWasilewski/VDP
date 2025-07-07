from utils_vdp import gaussian_loss, mc_nll, dirichlet_loss
from layers import MySoftmax
import wandb
from torch.nn import CrossEntropyLoss
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from models import VDPResNet18


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Randomly crop a 32x32 area with padding
    transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=128,
                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=128,
                         shuffle=False, num_workers=2)


class EarlyStopping:
    def __init__(self, patience=250, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model = None
        self.best_epoch = -1  # Track best epoch

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict()
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def ce_loss(mu, sigma, y):
    criterion = CrossEntropyLoss()
    return criterion(mu, y)

loss_list = ["gaussian", "mc", "dirichlet", "ce"]
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


for idx, loss in enumerate([gaussian_loss, mc_nll, dirichlet_loss, ce_loss]):
    kl_factor = 0.00001 if loss_list[idx] != "ce" else 0.0
    criterion = loss
    lr=0.001
    batch_size = 128
    num_epochs = 250

    config = {
    "learning_rate": lr,
    "loss_function": loss_list[idx],
    "kl_factor": kl_factor,
    "architecture": "Resnet18",
    "epochs": num_epochs,
    "batch_size": batch_size,
    "train_dataset": "CIFAR10",
    "device": device,
    }


    wandb.init(name = loss_list[idx] + "_kl000001_2", project="VDPResNet18_CIFAR10", config=config)  
    early_stopping = EarlyStopping(patience=1500, delta=0.001)
    
    net = VDPResNet18(num_classes=10)
    a = net(next(iter(testloader))[0])
    net = net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7)

    for epoch in range(num_epochs):
        log_dict = {}
        # Training phase
        net.train()
        running_loss, running_kl, correct, total, sigma_mean, sigma_std = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            mu, sigma, kl = net(inputs)
            if idx == 0:
                mu, sigma = MySoftmax()(mu, sigma)
            loss = criterion(mu, sigma, labels) + kl_factor * kl
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_kl += kl.item()        
            _, predicted = torch.max(mu, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            sigma_mean += sigma.mean(1).sum().item()
            sigma_std += sigma.std(1).sum().item()


        train_accuracy = 100 * correct / total
        train_loss = running_loss / total
        train_kl = running_kl / total
        sigma_mean /= total
        sigma_std /= total
        
        log_dict.update({
            "train/loss": train_loss,
            "train/kl_loss": train_kl,
            "train/accuracy": train_accuracy,
            "train/average_sigma": sigma_mean,
            "train/sigma_std": sigma_std,
        })

        net.eval()
        correct_test, running_kl_test, running_loss_test, total_test, test_sigma_sd, test_sigma_mean = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) # Move batch to GPU
            mu, sigma, kl = net(images)
            if idx == 0:
                mu, sigma = MySoftmax()(mu, sigma)
            loss = criterion(mu, sigma, labels) + kl_factor * kl
            
            running_loss_test += loss.item()
            running_kl_test += kl.item()
            _, predicted = torch.max(mu, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            test_sigma_mean += sigma.mean(1).sum().item()
            test_sigma_sd += sigma.std(1).sum().item()

        test_accuracy = 100 * correct_test / total_test
        test_loss = running_loss_test / total_test
        test_kl = running_kl_test / total_test
        test_sigma_mean = test_sigma_mean / total_test
        test_sigma_sd = test_sigma_sd / total_test

        scheduler.step(test_accuracy)
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]

        log_dict.update({
            "test/loss": test_loss,
            "test/kl_loss": test_kl,
            "test/accuracy": test_accuracy,
            "test/average_sigma": test_sigma_mean,
            "test/sigma_std": test_sigma_sd,
            "epoch": epoch + 1,
            "learning_rate": current_lr,
        })

        

        wandb.log(log_dict)

        
        early_stopping(test_loss, net, epoch + 1)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        if (epoch+1)%50  == 0:
            torch.save(net.state_dict(), f'models/resnet/CIFAR10/{loss_list[idx]}_{epoch+1}_kl000001_2.pth')
    if early_stopping.best_model is not None:
        best_epoch = early_stopping.best_epoch
        torch.save(early_stopping.best_model, f'models/resnet/CIFAR10/{loss_list[idx]}_best_epoch_{best_epoch}_kl000001_2.pth')

    wandb.finish()