import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import json
import numpy as np

# Assuming util_funx.py is in the same directory and has the load_state_dict fix
from utils.predict_utils import load_checkpoint

# --- Configuration ---
MODEL_PATH = "checkpoints/vit_b_16_model_checkpoint_7ep.pt"
CATEGORY_NAMES_PATH = "cat_to_name.json" # Set to None if you don't have this file
TOP_K = 5 # Number of top predictions to show

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model and Category Names ---
model = None
cat_to_name = None

try:
    print(f"Loading model from {MODEL_PATH}...")
    model = load_checkpoint(MODEL_PATH)
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print(f"Model loaded successfully and moved to {DEVICE}.")
    if not hasattr(model, 'class_to_idx'):
        raise AttributeError("Model loaded from checkpoint does not have 'class_to_idx' attribute.")
except FileNotFoundError:
    print(f"ERROR: Model checkpoint file not found at {MODEL_PATH}. Please ensure it's in the root directory.")
    # Gradio will show an error if the app launches with model=None
except AttributeError as e:
    print(f"ERROR: Model structure issue or 'class_to_idx' missing: {e}")
except Exception as e:
    print(f"ERROR: Could not load the model: {e}")

if CATEGORY_NAMES_PATH:
    try:
        with open(CATEGORY_NAMES_PATH, 'r') as f:
            cat_to_name = json.load(f)
        print(f"Category names loaded from {CATEGORY_NAMES_PATH}.")
    except FileNotFoundError:
        print(f"Warning: Category names file not found at {CATEGORY_NAMES_PATH}. Outputting class IDs.")
    except Exception as e:
        print(f"Error loading category names: {e}")

# --- Image Preprocessing (matches training) ---
preprocess_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Prediction Function for Gradio ---
def predict_flower(input_image_pil: Image.Image):
    if model is None:
        return {"Error": "Model not loaded. Please check server logs."}

    # Preprocess the PIL image
    img_tensor = preprocess_transform(input_image_pil)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
    img_tensor = img_tensor.to(DEVICE)

    # Get predictions
    with torch.no_grad():
        output = model.forward(img_tensor)
        probabilities = torch.exp(output) # Output is LogSoftmax, so take exp

    probs, indices = probabilities.topk(TOP_K)

    probs = probs.cpu().numpy().flatten()
    indices = indices.cpu().numpy().flatten()

    # Map indices to class IDs (original folder names)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_ids = [idx_to_class[idx] for idx in indices]

    # Map class IDs to actual names if cat_to_name is available
    if cat_to_name:
        class_labels = [cat_to_name.get(str(cid), f"Unknown ID: {cid}") for cid in class_ids]
    else:
        class_labels = [f"Class ID: {str(cid)}" for cid in class_ids]

    # Format for Gradio output (dictionary for 'label' component)
    results = {label: float(prob) for label, prob in zip(class_labels, probs)}
    return results

# --- Gradio Interface ---
iface = gr.Interface(
    fn=predict_flower,
    inputs=gr.Image(type="pil", label="Upload Flower Image"),
    outputs=gr.Label(num_top_classes=TOP_K, label="Predictions"),
    title="Flower Image Classifier",
    description="Upload an image of a flower to classify it. This model can identify up to 102 different flower types based on its training.",
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Launching Gradio interface...")
    iface.launch()