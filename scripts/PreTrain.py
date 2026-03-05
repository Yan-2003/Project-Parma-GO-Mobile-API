import sys
import os
import json
import torch
import warnings
import cv2
import numpy as np

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from rapidfuzz import process, fuzz
from wordfreq import zipf_frequency


warnings.filterwarnings("ignore")


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

        return [
            x.strip()
            for x in f
            if x.strip()
        ]


DRUGS = load_drugs()


# =============================
# DEVICE
# =============================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# =============================
# MODEL
# =============================

processor = TrOCRProcessor.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

model = (
    VisionEncoderDecoderModel
    .from_pretrained(MODEL_NAME)
    .to(device)
)

model.eval()


# =============================
# IMAGE QUALITY
# =============================

def blur_score(img):

    gray = cv2.cvtColor(
        np.array(img),
        cv2.COLOR_RGB2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


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


    # Remove weird chars
    clean = "".join(
        c for c in text
        if c.isalnum() or c.isspace()
    ).strip()


    # Reject garbage
    if len(clean) < 4 or not any(c.isalpha() for c in clean):
        return "", 0


    words = clean.split()

    best_name = ""
    best_score = 0


    # Try word + phrase
    candidates = words + [clean]


    for cand in candidates:

        match = process.extractOne(
            cand,
            DRUGS,
            scorer=fuzz.ratio
        )

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

    return sum(
        zipf_frequency(w, "en")
        for w in words
    )


# =============================
# OCR
# =============================

def run_ocr(img):

    results = []


    for angle in [0, 90, 180, 270]:

        rotated = img.rotate(angle, expand=True)

        pixel = processor(
            rotated,
            return_tensors="pt"
        ).pixel_values.to(device)


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


        text = processor.batch_decode(
            out.sequences,
            skip_special_tokens=True
        )[0].strip()


        score = 0

        if out.scores:
            score = sum(
                s.max().item()
                for s in out.scores
            )


        results.append((text, score))


    return results


# =============================
# MAIN
# =============================

try:

    if len(sys.argv) < 2:
        raise ValueError("Missing image path")


    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        raise ValueError("Image not found")


    img = Image.open(image_path)

    img = preprocess(img)


    # Blur filter
    clarity = blur_score(img)

    if clarity < BLUR_THRESHOLD:

        print(json.dumps({
            "text": "",
            "warning": "Image too blurry",
            "clarity": round(clarity, 2)
        }))

        sys.exit(0)


    # OCR
    results = run_ocr(img)


    candidates = []


    for text, conf in results:

        # Reject junk OCR
        if not text or not any(c.isalpha() for c in text):
            continue

        if conf < CONF_THRESHOLD:
            continue


        corr, match_score = correct_drug(text)

        if not corr:
            continue


        lscore = lang_score(corr)


        final_score = (
            conf * 0.6 +
            match_score * 0.3 +
            lscore * 0.1
        )


        candidates.append((
            corr,
            final_score,
            conf,
            match_score,
            lscore
        ))


    # No valid result
    if not candidates:

        print(json.dumps({
            "text": "",
            "warning": "Low confidence",
            "raw_results": results
        }))

        sys.exit(0)


    # Pick best
    best = sorted(
        candidates,
        key=lambda x: x[1],
        reverse=True
    )[0]


    print(json.dumps({
        "text": best[0],
        "score": round(best[1], 2),
        "raw_conf": round(best[2], 2),
        "drug_match": best[3],
        "lang_score": round(best[4], 2),
        "clarity": round(clarity, 2),
        "drugs_loaded": len(DRUGS)
    }))


except Exception as e:

    print(json.dumps({
        "error": str(e)
    }))

    sys.exit(2)