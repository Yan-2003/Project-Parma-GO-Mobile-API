from ultralytics import YOLO

model = YOLO(r"S:\Project Pharma\Pharma YoloV8\runs\detect\pharma_OCR2\weights\best.pt")
results = model.predict(source=r"S:\Project Pharma\Pharma API\uploads\amoxicillin.jpg", show=True)
