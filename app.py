from flask import Flask, request, jsonify, render_template
# from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from helper import pred_and_plot_image
import logging
from typing import Tuple

app = Flask(__name__)

def build_effnet_b0(num_classes: int = 3) -> torch.nn.Module:
    """Create EfficientNet-B0 with custom classifier for num_classes."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_trained_model(model_path: str, num_classes: int = 3) -> torch.nn.Module:
    """Load a model checkpoint that may be a full model or a state_dict.

    Tries the following in order:
    - If checkpoint is an nn.Module: return it as-is.
    - If checkpoint is a mapping of weights: build EfficientNet-B0 head for num_classes and load state_dict (strict=False).
    Supports common keys: model_state_dict, state_dict, or a raw state_dict mapping.
    """
    ckpt = torch.load(model_path, map_location=torch.device("cpu"))

    # Case 1: full model object
    if isinstance(ckpt, torch.nn.Module):
        return ckpt

    # Case 2: dictionary-like -> get state_dict
    state_dict = None
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Heuristic: looks like a state_dict if keys look like layer names
            if all(isinstance(k, str) for k in ckpt.keys()):
                state_dict = ckpt

    if state_dict is None:
        raise RuntimeError("Unsupported checkpoint format: expected nn.Module or (state_)dict")

    model = build_effnet_b0(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logging.warning("State dict load mismatches. Missing: %s | Unexpected: %s", missing, unexpected)
    return model


# --- Load model ---
MODEL_PATH = "model/pizza_steak_sushi_effnetb0.pth"
CLASS_NAMES = ["pizza", "steak", "sushi"]
model = load_trained_model(MODEL_PATH, num_classes=len(CLASS_NAMES))


# --- Image preprocessing (adjust for your model) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image field in form"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class_names = CLASS_NAMES
    pred_idx, probs = pred_and_plot_image(
        model=model,
        image_path=file,
        class_names=class_names,
        transform=manual_transforms,
    )

    prob = float(probs[0, pred_idx].item())
    label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)

    return jsonify({
        "index": int(pred_idx),
        "label": label,
        "confidence": prob,
    })

if __name__ == "__main__":
    # Use 0.0.0.0 to expose it to the local network
    app.run(host="0.0.0.0", port=5000, debug=True)
