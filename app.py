from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Plant Model API running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    temp_path = "temp_image.png"
    file.save(temp_path)  # save uploaded image temporarily

    try:
        # Run your existing predict.py script like in terminal
        result = subprocess.run(
            ["python", "predict.py", temp_path],
            capture_output=True,
            text=True
        )
        os.remove(temp_path)  # remove temp image

        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500

        # predict.py prints: Prediction: <class_name>
        output_line = result.stdout.strip().split(":")[-1].strip()
        return jsonify({"prediction": output_line})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)