import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Setup the device configuration 
device = torch.vision('cuda' if torch.cuda.is_available() else 'cpu')

# train data loader
def get_train_valid_loader(data_dir, batch_size, augment, random_seed, valid_size = 0.1, shuffle = True):
    
    normalize = transforms.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010],
    )
    
    # normalize = transforms.Normalize(
    #     mean = [0.485, 0.456, 0.406],
    #     std = [0.229, 0.224, 0.225],
    # )
    
    val_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
    ])
    
    if augment:
        train_transform = transforms.compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    else:
        train_transform = transforms.compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])
        
        
# Load dataset
    train_dataset = datasets.ImageFolder(root = "path", transform = train_transform)
    val_dataset = datasets.ImageFolder(root = "path", transform = val_transform)
    
    
# creating the dataloader
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    valid_loader = DataLoader(val_dataset, batch_size = batch_size)

# test data loader 
def get_test_loader(data_dir, batch_size, shuffle = True):
    
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225], 
    )
    
    test_transform  = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_dataset = datasets.ImageFolder(root = "path", transform = test_transform)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
    
    
#  Local response Normalization 

class LRN(nn.Module):
    def __init__(self, size, alpha = 1e-4, beta = 0.75, k =2):
        super(LRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
    
    def forward(self, x):
        div = x.pow(2).unsqueeze(1)
        div = F.avg_pool3d(div, (self.size, 1, 1), stride = 1, padding = (self.size // 2, 0, 0))
        div = div.squeeze(1)
        div = div*self.alpha
        div = div + self.k
        div = div.pow(self.beta)
        x = x/div
        return x
    
    
# Alexnet architecture 
class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(AlexNet, self).__init__()
        
        
            # conv layer 1
        self.layer1 = nn.sequential(
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 2)
            nn.ReLU(inplace = True),
            LRN(size = 5),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            )
        # conv layer 2 
        nn.Conv2d(96, 256, kernel_size = 5, padding = 2),
        nn.ReLU(inplace = True),
        LRN(size = 5),
        nn.MaxPool2d(kernel_size = 3, stride = 2),
        # conv layer 3 
        nn.Conv2d(256, 384, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        # conv layer 4
        nn.Conv2d(384, 384, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifiers(x)
        return x
    
                  
        

        