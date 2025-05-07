import torch.nn as nn
from layers import *

class VGG11_224(nn.Module):
    def __init__(self, num_classes=1000, bias=True):
        super(VGG11_224, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=bias), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=bias), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=bias), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=bias), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=bias), nn.ReLU(),
            nn.Linear(4096, 4096, bias=bias), nn.ReLU(),
            nn.Linear(4096, num_classes, bias=bias)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


#  -------------------- VDP models -------------------- 

class VDP_VGG11_224(nn.Module):
    def __init__(self, kernel_sizes=[64, 128, 256, 512], num_classes=1000, padding=1):
        super(VDP_VGG11_224, self).__init__()

        self.pool = VDPMaxPooling()

        self.conv1_1 = VDPFirstConv(kernel_size=3, kernel_num=kernel_sizes[0], kernel_stride=1, padding=padding)
        
        self.conv2_1 = VDPIntermediateConv(kernel_size=3, kernel_num=kernel_sizes[1], kernel_stride=1, padding=padding)
        
        self.conv3_1 = VDPIntermediateConv(kernel_size=3, kernel_num=kernel_sizes[2], kernel_stride=1, padding=padding)
        self.conv3_2 = VDPIntermediateConv(kernel_size=3, kernel_num=kernel_sizes[2], kernel_stride=1, padding=padding)
        
        self.conv4_1 = VDPIntermediateConv(kernel_size=3, kernel_num=kernel_sizes[3], kernel_stride=1, padding=padding)
        self.conv4_2 = VDPIntermediateConv(kernel_size=3, kernel_num=kernel_sizes[3], kernel_stride=1, padding=padding)
        
        self.conv5_1 = VDPIntermediateConv(kernel_size=3, kernel_num=kernel_sizes[3], kernel_stride=1, padding=padding)
        self.conv5_2 = VDPIntermediateConv(kernel_size=3, kernel_num=kernel_sizes[3], kernel_stride=1, padding=padding)

        self.fc1 = VDPIntermediateLinear(4096)
        self.fc2 = VDPIntermediateLinear(4096)
        self.fc3 = VDPIntermediateLinear(num_classes)

        self.relu = VDP_ReLU()

    def forward(self, inputs, training=True):
        batch_size = inputs.size(0)

        mu, sigma, kl_0 = self.conv1_1(inputs)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma, kl_1 = self.conv2_1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)
        
        mu, sigma, kl_2 = self.conv3_1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma, kl_3 = self.conv3_2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)
        
        mu, sigma, kl_4 = self.conv4_1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma, kl_5 = self.conv4_2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma, kl_6 = self.conv5_1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma, kl_7 = self.conv5_2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu = mu.view(batch_size, -1)
        sigma = sigma.view(batch_size, -1)
        mu1, sigma, kl_8 = self.fc1(mu, sigma)
        mu2, sigma = self.relu(mu1, sigma)
        mu3, sigma, kl_9 = self.fc2(mu2, sigma)
        mu4, sigma = self.relu(mu3, sigma)
        mu_out5, sigma_out, kl_10 = self.fc3(mu4, sigma)
        
        kl = kl_0 + kl_1 + kl_2 + kl_3 + kl_4 + kl_5 + kl_6 + kl_7 + kl_8 + kl_9 + kl_10

        return mu_out5, sigma_out, kl

    def get_parameters(self):
        return list([p for n,p in self.named_parameters() if 'mu' in n])

    def get_parameters_sigma(self):
        return list([p for n,p in self.named_parameters() if 'sigma' in n])

    def load_weights(self, model):
        vdp_params = list(self.parameters())
        model_params = list(model.parameters())
        j = 0
        for i in range(0, len(vdp_params), 2):
            # Ensure that the shapes match before copying data
            if vdp_params[i].shape == model_params[j].shape:
                vdp_params[i].data.copy_(model_params[j].data)
                j += 1
            # elif len(vdp_params[i].shape) == 2 and len(model_params[j].shape) == 2 and \
            #     vdp_params[i].shape[0] == model_params[j].shape[1] and \
            #     vdp_params[i].shape[1] == model_params[j].shape[0]:
            #     vdp_params[i].data.copy_(model_params[j].data.t()) # handle linear layer 
            #     j += 1
            else:
                print(
                    f"Shape mismatch at parameter {i}. Expected shape {vdp_params[i].shape}, "
                    f"got {model_params[j].shape}"
                )
                return  # Stop if there's a shape mismatch
        print("Parameters assigned successfully.")


class VDPResNet18(nn.Module):
    def __init__(self, num_classes=101):
        super(VDPResNet18, self).__init__()
        self.in_channels = 64
                
        # Initial convolution layer and maxpool
        self.conv1 = VDPFirstConv(kernel_size=7, kernel_num=64, kernel_stride=2, padding=3)
        self.bn1 = VDP_BatchNorm2D()
        self.relu = VDP_ReLU()
        self.maxpool = VDPMaxPooling(pooling_size=3, pooling_stride=2, pooling_pad=1)      
        
        
        # ResNet layers
        self.layer11 = VDPResnetBasicBlock(64, stride1=1, stride2=1, previous_channels=64)
        self.layer12 = VDPResnetBasicBlock(64, stride1=1, stride2=1, previous_channels=64)

        self.layer21 = VDPResnetBasicBlock(128, stride1=1, stride2=1, previous_channels=64)
        self.layer22 = VDPResnetBasicBlock(128, stride1=2, stride2=1, previous_channels=128)
        
        self.layer31 = VDPResnetBasicBlock(256, stride1=1, stride2=1, previous_channels=128)
        self.layer32 = VDPResnetBasicBlock(256, stride1=2, stride2=1, previous_channels=125)
        
        self.layer41 = VDPResnetBasicBlock(512, stride1=1, stride2=1, previous_channels=256)
        self.layer42 = VDPResnetBasicBlock(512, stride1=2, stride2=1, previous_channels=512)
    
        self.gap = VDP_GAP()

        self.fc = VDPIntermediateLinear(num_classes)
    
    def forward(self, x):
        mu, sigma, kl = self.conv1(x)
        mu, sigma = self.bn1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.maxpool(mu, sigma)

        mu, sigma, kl1 = self.layer11(mu, sigma)
        mu, sigma, kl2 = self.layer12(mu, sigma)
        mu, sigma, kl3 = self.layer21(mu, sigma)
        mu, sigma, kl4 = self.layer22(mu, sigma)
        mu, sigma, kl5 = self.layer31(mu, sigma)
        mu, sigma, kl6 = self.layer32(mu, sigma)
        mu, sigma, kl7 = self.layer41(mu, sigma)
        mu, sigma, kl8 = self.layer42(mu, sigma)
        mu, sigma = self.gap(mu, sigma)
        mu = torch.flatten(mu, 1)
        sigma = torch.flatten(sigma, 1)
        mu, sigma, kl = self.fc(mu, sigma)
        kl = kl + kl1 + kl2 + kl3 + kl4 + kl5 + kl6 + kl7 + kl8
        return mu, sigma, kl
    




# VDP_MLP
# VDP_VGG11_32
# VDP_resnet50
# VDP_widenet