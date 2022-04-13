import torch
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.CIFAR10(root="./Data",train = True , transform = transform , download=True)
val_data = datasets.CIFAR10(root="./Data",train = False , transform = transform , download=True)

print(train_data)
print(val_data)


train_loader = DataLoader(train_data,batch_size=1000,shuffle=True)
val_loader = DataLoader(val_data,batch_size=1000,shuffle=True)