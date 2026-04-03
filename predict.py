# predict.py
import torch
from PIL import Image
from utils import get_transforms
from model import PlantModel
from classes import classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model.pth"



model = PlantModel()
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device)
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms()
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return classes[predicted.item()]

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]
    disease_name = predict(img_path)
    print(f"Predicted Disease: {disease_name}")