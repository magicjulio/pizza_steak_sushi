# Pizza/Steak/Sushi Classifier (Flask)

Minimal Flask app to upload an image and get a prediction from a PyTorch model.

## Setup

1. Create a virtual environment (optional but recommended).
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Ensure the model file exists at `model/pizza_steak_sushi_effnetb0.pth`.

## Run

```
python app.py
```

Open http://localhost:5000 and upload an image.

## Notes
- The app expects RGB images and resizes to 224x224 with ImageNet normalization.
- Returned JSON from `/predict`:
  - `index`: predicted class index
  - `label`: human-friendly label
  - `confidence`: softmax probability for the predicted class (0-1)
