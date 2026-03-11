import os
import sys
import json
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from rapidfuzz import process, fuzz
import re

# =============================
# CONFIG
# =============================

MODEL_NAME = "microsoft/trocr-base-handwritten"

BLUR_THRESHOLD = 60
MIN_MATCH = 65
MAX_LEN = 64

ROTATIONS = [0, 90, 180, 270]

# =============================
# LOAD DRUG LIST
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRUG_FILE = os.path.join(BASE_DIR, "drugs.txt")

def load_drugs():
    if not os.path.exists(DRUG_FILE):
        return []

    with open(DRUG_FILE, "r", encoding="utf-8") as f:
        return [x.strip().lower() for x in f if x.strip()]

DRUGS = load_drugs()

# =============================
# DEVICE
# =============================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# MODEL
# =============================

processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# =============================
# IMAGE QUALITY
# =============================

def blur_score(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# =============================
# PREPROCESS
# =============================

def preprocess(img):

    img = ImageOps.exif_transpose(img)

    img = img.convert("L")

    img = np.array(img)

    # CLAHE contrast improvement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # adaptive threshold
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    # noise removal
    kernel = np.ones((2,2),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    img = Image.fromarray(img)

    img = ImageEnhance.Sharpness(img).enhance(2.5)

    return img.convert("RGB")

# =============================
# CLEAN OCR TEXT
# =============================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-z]', '', text)

    return text

# =============================
# OCR
# =============================

def run_ocr(img):

    results = []

    for angle in ROTATIONS:

        rotated = img.rotate(angle, expand=True)

        pixel = processor(rotated, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():

            output = model.generate(
                pixel,
                max_length=MAX_LEN,
                num_beams=5,
                return_dict_in_generate=True,
                output_scores=True
            )

        text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]

        text = clean_text(text)

        if text:
            results.append(text)

    return results

# =============================
# MATCH DRUG
# =============================

def match_drug(ocr_results):

    best_drug = ""
    best_score = 0

    for text in ocr_results:

        match = process.extractOne(
            text,
            DRUGS,
            scorer=fuzz.ratio
        )

        if match:

            name, score, _ = match

            # combine multiple fuzzy metrics
            partial = fuzz.partial_ratio(text, name)
            token = fuzz.token_set_ratio(text, name)

            final_score = (score * 0.5) + (partial * 0.3) + (token * 0.2)

            if final_score > best_score:

                best_score = final_score
                best_drug = name

    if best_score >= MIN_MATCH:
        return best_drug

    return ""

# =============================
# MAIN
# =============================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing image path"}))
        sys.exit(2)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(json.dumps({"error": "Image not found"}))
        sys.exit(2)

    img = preprocess(Image.open(image_path))

    clarity = blur_score(img)

    if clarity < BLUR_THRESHOLD:

        print(json.dumps({
            "text": "",
            "warning": "Image too blurry"
        }))

        sys.exit(0)

    ocr_results = run_ocr(img)

    best_match = match_drug(ocr_results)

    print(json.dumps({
        "text": best_match
    }))