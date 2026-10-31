import torch.nn as nn

# Fully Connected Neural Network (FCNN)
class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Enhanced CNN (with BatchNorm + Dropout)
class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------- GAN (Module 6 / Assignment 3) ----------

# Generator: noise (N, 100) -> image (N, 1, 28, 28) in [-1, 1]
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)                                     # (N, 100) -> (N, 6272)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 7x7 -> 14x14
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)    # 14x14 -> 28x28
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc(z.view(z.size(0), -1))
        x = x.view(-1, 128, 7, 7)
        x = self.relu(self.bn1(self.deconv1(x)))
        return self.tanh(self.deconv2(x))

# Discriminator: image (N, 1, 28, 28) -> one logit per image (apply sigmoid for real/fake probability)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)    # 28x28 -> 14x14
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 14x14 -> 7x7
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.bn2(self.conv2(x)))
        return self.fc(self.flatten(x))

# Wrapper so get_model("GAN") returns both networks as one module
class GAN(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()

    def forward(self, z):
        return self.generator(z)

# Factory function to get model by name
def get_model(model_name):
    if model_name == "FCNN":
        return FCNN()
    elif model_name == "CNN":
        return CNN()
    elif model_name == "EnhancedCNN":
        return EnhancedCNN()
    elif model_name == "GAN":
        return GAN()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
