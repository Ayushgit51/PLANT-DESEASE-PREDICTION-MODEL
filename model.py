# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet34

class PlantModel(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantModel, self).__init__()
        self.model = resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(path, device='cpu', num_classes=38):
    model = PlantModel(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model