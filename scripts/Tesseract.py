import sys
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if len(sys.argv) < 2:
    print("Usage: python OCR_MODEL.py <image_path>", file=sys.stderr)
    sys.exit(1)

image_path = sys.argv[1]

try:
    img = Image.open(image_path)
    img = img.convert('L')  # Grayscale
    img = img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3)  # Increase contrast

    # Binarize image
    img = img.point(lambda x: 0 if x < 140 else 255, '1')

    # Use single character mode
    text = pytesseract.image_to_string(img, config='--psm 7')
    print(text.strip())
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(2)