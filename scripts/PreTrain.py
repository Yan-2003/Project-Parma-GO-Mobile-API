import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from rapidfuzz import process, fuzz
from wordfreq import zipf_frequency

# =============================
# CONFIG
# =============================
MODEL_NAME = "microsoft/trocr-base-handwritten"
CONF_THRESHOLD = -20
BLUR_THRESHOLD = 70
MIN_MATCH = 65
MAX_LEN = 64

# =============================
# LOAD DRUG LIST
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRUG_FILE = os.path.join(BASE_DIR, "drugs.txt")

def load_drugs():
    if not os.path.exists(DRUG_FILE):
        return []
    with open(DRUG_FILE, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

DRUGS = load_drugs()

# =============================
# DEVICE
# =============================
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = ImageEnhance.Sharpness(img).enhance(2.3)
    img = img.filter(ImageFilter.MedianFilter(3))
    return img.convert("RGB")

# =============================
# SMART DRUG CORRECTION
# =============================
def correct_drug(text):
    if not DRUGS:
        return "", 0

    clean = "".join(c for c in text if c.isalnum() or c.isspace()).strip()
    if len(clean) < 4 or not any(c.isalpha() for c in clean):
        return "", 0

    words = clean.split()
    best_name = ""
    best_score = 0
    candidates = words + [clean]

    for cand in candidates:
        match = process.extractOne(cand, DRUGS, scorer=fuzz.ratio)
        if not match:
            continue
        name, score, _ = match
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= MIN_MATCH:
        return best_name, best_score

    return "", best_score

# =============================
# LANGUAGE SCORE
# =============================
def lang_score(text):
    words = text.lower().split()
    return sum(zipf_frequency(w, "en") for w in words)

# =============================
# OCR
# =============================
def run_ocr(img):
    results = []
    for angle in [0, 90, 180, 270]:
        rotated = img.rotate(angle, expand=True)
        pixel = processor(rotated, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            out = model.generate(
                pixel,
                max_length=MAX_LEN,
                num_beams=2,
                do_sample=False,
                repetition_penalty=1.4,
                length_penalty=0.9,
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        score = sum(s.max().item() for s in out.scores) if out.scores else 0
        results.append((text, score))
    return results

# =============================
# OPTIONAL CLI USAGE
# =============================
""" if __name__ == "__main__":
    import sys, json
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
        print(json.dumps({"text": "", "warning": "Image too blurry", "clarity": round(clarity,2)}))
        sys.exit(0)
    results = run_ocr(img)
    print(json.dumps({"results": results})) """

if __name__ == "__main__":
    import sys, json

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
            "warning": "Image too blurry",
            "clarity": round(clarity,2)
        }))
        sys.exit(0)

    results = run_ocr(img)

    best_text = ""
    best_score = -999999

    for text, score in results:

        clean = text.strip()

        # ignore empty
        if len(clean) < 3:
            continue

        # ignore numeric garbage like "1 1"
        if not any(c.isalpha() for c in clean):
            continue

        if score > best_score:
            best_score = score
            best_text = clean

    # fallback if all filtered
    if best_text == "" and results:
        best_text = results[0][0]

    # drug dictionary correction
    corrected, match_score = correct_drug(best_text)

    final_text = corrected if corrected else best_text

    print(json.dumps({
        "text": final_text.lower().strip()
    }))