import torch
import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):
    """
    Simple CNN Architecture: MesoNet (Meso-4)
    Focuses on the "mesoscopic" properties of images (noise analysis) 
        rather than high-level semantic features. It is significantly smaller 
        than modern networks.
    
    Architecture:
        - Input: 256x256x3
        - Conv Block 1:  8 filters (3x3), Batch Norm, ReLU, MaxPool (2x2)
        - Conv Block 2:  8 filters (3x3), Batch Norm, ReLU, MaxPool (2x2)
        - Conv Block 3: 16 filters (5x5), Batch Norm, ReLU, MaxPool (2x2)
        - Conv Block 4: 16 filters (5x5), Batch Norm, ReLU, MaxPool (2x2)
        - Classifer: Flatten -> FC (16 units) -> LeakyReLU -> Dropout -> FC (1 unit) -> Sigmoid
    
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Block 1.
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8,
            kernel_size=3, padding=1, bias=False)
        self. bn1 = nn.BatchNorm2d(num_features=8)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2.
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=8,
            kernel_size=3, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        
        # Block 3.
        self.conv3 = nn.Conv2d(
            in_channels=8, out_channels=16,
            kernel_size=5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        
        # Block 4.
        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=16,
            kernel_size=5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=16)
        
        # Classifer
        self.flatten_dim = 16 * 16 * 16
        self.fc1 = nn.Linear(in_features=self.flatten_dim, out_features=16)
        self.leakly_relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(p=0.5)
        
        # Output
        self.fc2 = nn.Linear(in_features=16, out_features=num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x) # [8, 128, 128]
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x) # [8, 64, 64]
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x) # [16, 32, 32]
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x) # [16, 16, 16]
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.leakly_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    


class EfficientNetDF(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetDF, self).__init__()
        
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        
        # replace classifer
        num_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes))
        
    def forward(self, x):
        return self.model(x)