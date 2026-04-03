import torch.nn as nn
from torchvision import models

class PlantModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantModel, self).__init__()
        
        self.model = models.resnet50(weights=None)  # keep ResNet50
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)