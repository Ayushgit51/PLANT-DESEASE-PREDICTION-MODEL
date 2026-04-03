import os
import torch
import gc
from flask import Flask, request, jsonify
from PIL import Image
from utils import get_transform
from classes import class_names

app = Flask(__name__)

# 🔥 MEMORY OPTIMIZATION
torch.set_num_threads(1)
torch.set_grad_enabled(False)

model = None  # lazy loading
transform = get_transform()

def load_model():
    global model
    if model is None:
        from model import PlantModel

        model = PlantModel(num_classes=len(class_names))

        state_dict = torch.load("model.pth", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        del state_dict
        gc.collect()


@app.route("/")
def home():
    return "Plant Disease API Running"


@app.route("/predict", methods=["POST"])
def predict():
    load_model()  # 🔥 load only when needed

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return jsonify({
        "prediction": class_names[predicted.item()],
        "confidence": float(confidence.item())
    })


# 🔥 REQUIRED FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)