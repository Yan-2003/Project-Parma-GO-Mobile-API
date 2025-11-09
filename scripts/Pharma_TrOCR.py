import sys
import os
import json
import warnings
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../Models/TrOCR/results/checkpoint-15600"
)

# Verify model folder exists
if not os.path.exists(MODEL_PATH):
    print(json.dumps({"error": f"Model not found: {MODEL_PATH}"}))
    sys.exit(1)

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)

# Check for image path
if len(sys.argv) < 2:
    print(json.dumps({"error": "Usage: python Pharma_TrOCR.py <image_path>"}))
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(json.dumps({"error": f"File not found: {image_path}"}))
    sys.exit(1)

try:
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocess and predict
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(json.dumps({"text": text.strip()}))

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(2)