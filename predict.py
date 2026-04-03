import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from utils import get_transform
from classes import class_names

# ✅ load model (same as training: ResNet50)
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ✅ load image
image_path = "Test/leaf.png"
image = Image.open(image_path).convert("RGB")

transform = get_transform()
image = transform(image).unsqueeze(0)

# ✅ prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Prediction:", class_names[predicted.item()])