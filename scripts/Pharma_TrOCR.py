import sys
import os
import json
import warnings
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../models/TrOCR/model"
)

# Verify model folder exists
if not os.path.exists(MODEL_PATH):
    print(json.dumps({"error": f"Model not found: {MODEL_PATH}"}))
    sys.exit(1)

# Device selection (CUDA > MPS > CPU)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

# Check for image path
if len(sys.argv) < 2:
    print(json.dumps({"error": "Usage: python Pharma_TrOCR.py <image_path>"}))
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(json.dumps({"error": f"File not found: {image_path}"}))
    sys.exit(1)


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Improve image quality for OCR
    """
    # Fix EXIF orientation (important for phone cameras)
    img = ImageOps.exif_transpose(img)

    # Convert to grayscale
    img = img.convert("L")

    # Increase contrast
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # Increase sharpness
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    # Light denoise
    img = img.filter(ImageFilter.MedianFilter(size=3))

    # Optional binarization
    img = img.point(lambda x: 0 if x < 140 else 255, "1")

    return img.convert("RGB")


def ocr_with_rotation(img: Image.Image):
    """
    Try OCR at multiple rotations and pick best result
    """
    best_text = ""
    best_score = -float("inf")

    for angle in [0, 90, 180, 270]:
        rotated = img.rotate(angle, expand=True)

        pixel_values = processor(rotated, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            output = model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True
            )

        text = processor.batch_decode(
            output.sequences,
            skip_special_tokens=True
        )[0].strip()

        # Simple confidence score using log-probability
        score = sum(
            s.max().item() for s in output.scores
        ) if output.scores else 0

        if score > best_score and len(text) > 0:
            best_score = score
            best_text = text

    return best_text


try:
    image = Image.open(image_path)

    # Preprocess
    image = preprocess_image(image)

    # OCR with auto-rotation
    text = ocr_with_rotation(image)

    print(json.dumps({
        "text": text
    }))

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(2)