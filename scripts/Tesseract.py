import sys
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if len(sys.argv) < 2:
    print("Usage: python OCR_MODEL.py <image_path>", file=sys.stderr)
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"Error: File not found -> {image_path}", file=sys.stderr)
    sys.exit(1)

try:
    img = Image.open(image_path)
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    threshold = 140
    img = img.point(lambda x: 0 if x < threshold else 255, "1")
    config = "--psm 7"
    text = pytesseract.image_to_string(img, config=config)
    if text.strip():
        print(text.strip())
    else:
        print("No text detected.", file=sys.stderr)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(2)
