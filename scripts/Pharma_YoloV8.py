from ultralytics import YOLO
import sys
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Load your custom handwriting model
model = YOLO(r"S:\Project Pharma\Pharma YoloV8\runs\detect\pharma_OCR2\weights\best.pt")

# Map class indices to your actual characters/words
# Example: {0: "Paracetamol", 1: "Ibuprofen", 2: "500mg", ...}
class_map = {
    0: "Amoxicillin"
}

if len(sys.argv) < 2:
    print(json.dumps({"error": "Usage: python OCR_MODEL.py <image_path>"}))
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(json.dumps({"error": f"File not found: {image_path}"}))
    sys.exit(1)

try:
    # Predict (show=False avoids GUI)
    results = model.predict(source=image_path, show=False, conf=0.3)

    detected_texts = []

    # YOLO results per image
    for result in results:
        # Each detected box
        for box, cls in zip(result.boxes, result.boxes.cls):
            cls_idx = int(cls.item())
            text = class_map.get(cls_idx, "")
            if text:
                detected_texts.append(text)

    final_text = ' '.join(detected_texts)

    print(json.dumps({"text": final_text.strip()}))

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(2)
