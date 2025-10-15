import sys
import os
import easyocr
import warnings
import torch

warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    print("Usage: python OCR_MODEL.py <image_path>", file=sys.stderr)
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"Error: Image file not found -> {image_path}", file=sys.stderr)
    sys.exit(1)

try:
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    results = reader.readtext(image_path)
    
    if not results:
        print("Warning: No text detected in image.", file=sys.stderr)
    
    text = ' '.join([res[1] for res in results])
    print(text.strip())

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(2)
