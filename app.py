import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from utils import get_transform
from classes import class_names

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pth"
model = None

# ✅ Load model only if exists
if os.path.exists(MODEL_PATH):
    print("Loading model...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
else:
    print("⚠️ model.pth not found. Upload it after deployment.")

transform = get_transform()

# ✅ Health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API running", "model_loaded": model is not None})

# ✅ Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not found. Please upload model.pth on server."
        }), 500

    try:
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)