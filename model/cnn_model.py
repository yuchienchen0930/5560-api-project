import os
# Fix: Anaconda and PyTorch each ship their own OpenMP runtime (libiomp5md.dll),
# which crashes the kernel (OMP Error #15) when matplotlib is used. Must run before importing torch.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assignment 2 Part 1: CNN matching the required architecture (64x64x3 input, 10 classes)
class AssignmentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AssignmentCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)   # 64x64x3  -> 64x64x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                  # 64x64x16 -> 32x32x16

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 32x32x16 -> 32x32x32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                  # 32x32x32 -> 16x16x32

        self.fc1 = nn.Linear(32 * 16 * 16, 100)                             # 8192 -> 100
        self.fc2 = nn.Linear(100, num_classes)                              # 100 -> 10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# MNIST digit classifier (earlier version, still used by the /predict-digit endpoint)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
