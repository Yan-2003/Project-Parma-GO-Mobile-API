import sys
import os
import easyocr
import warnings
import torch
import cv2
import numpy as np
import json

warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    print(json.dumps({"error": "Usage: python OCR_MODEL.py <image_path>"}))
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(json.dumps({"error": f"File not found: {image_path}"}))
    sys.exit(1)

try:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_percent = 200
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )

    preprocessed_path = "preprocessed_temp.png"
    cv2.imwrite(preprocessed_path, gray)

    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    results = reader.readtext(
        preprocessed_path,
        detail=1,
        paragraph=True,
        contrast_ths=0.05,
        adjust_contrast=0.7,
        text_threshold=0.2,
        low_text=0.1
    )

    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    if not results:
        print(json.dumps({"text": ""}))
        sys.exit(0)

    text = ' '.join([res[1] for res in results])
    print(json.dumps({"text": text.strip()}))

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(2)
