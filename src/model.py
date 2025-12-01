import torch
import torch.nn as nn
from torchvision import models

class EffcientNetDF(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(EffcientNetDF, self).__init__()
        
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        
        # replace classifer
        num_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=num_features, out_features=num_classes),
            nn.Sigmoid())
        
    def forward(self, x):
        return self.model(x)