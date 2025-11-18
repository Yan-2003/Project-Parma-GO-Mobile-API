import os
import pandas as pd
from PIL import Image
from datasets import Dataset
import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================
# 📂 CORRECT PATH SETUP
# ============================================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TEST_DIR = os.path.join(BASE_DIR, "Testing")
TEST_CSV = os.path.join(TEST_DIR, "testing_labels.csv")
TEST_IMG_DIR = os.path.join(TEST_DIR, "testing_words")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# ============================================
# 📌 LOAD DATASET
# ============================================
def load_dataset(csv_path, image_dir):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV missing → {csv_path}")

    df = pd.read_csv(csv_path)

    # Required columns
    required_cols = {"IMAGE", "MEDICINE_NAME", "GENERIC_NAME"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain: {required_cols}")

    # Build image path + text label
    df["image_path"] = df["IMAGE"].apply(lambda x: os.path.join(image_dir, x))
    df["text"] = df["MEDICINE_NAME"].astype(str) + " " + df["GENERIC_NAME"].astype(str)

    return Dataset.from_pandas(df[["image_path", "text"]])


# ============================================
# ⚙️ LOAD PROCESSOR + MODEL
# ============================================
print("📂 Loading trained model and processor...")

if not os.path.exists(MODEL_DIR):
    print(f"❌ Model directory not found at {MODEL_DIR}")
    print("Please ensure the model has been trained first using train.py")
    exit(1)

try:
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    print("✅ Model and processor loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"📱 Using device: {device}")

# ============================================
# 🧠 PREPROCESS FUNCTION
# ============================================
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")

    pixel_values = processor(
        image,
        return_tensors="pt"
    ).pixel_values.squeeze(0)

    labels = processor.tokenizer(
        example["text"],
        max_length=64,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).input_ids[0]

    return {"pixel_values": pixel_values, "labels": labels, "text": example["text"]}

# ============================================
# 📊 LOAD AND PREPROCESS TEST DATASET
# ============================================
print("📂 Loading test dataset...")
test_dataset = load_dataset(TEST_CSV, TEST_IMG_DIR)
print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

print("⚙️ Preprocessing test data...")
test_dataset = test_dataset.map(preprocess)

# Set the format to pytorch tensors
test_dataset.set_format("torch")

# ============================================
# 🧪 EVALUATION METRICS
# ============================================
def calculate_cer(predictions, ground_truth):
    """Calculate Character Error Rate (CER)"""
    errors = 0
    total_chars = 0
    
    for pred, truth in zip(predictions, ground_truth):
        # Normalize strings
        pred = str(pred).strip()
        truth = str(truth).strip()
        
        # Calculate edit distance (simplified)
        total_chars += len(truth)
        if pred != truth:
            errors += max(len(pred), len(truth))
    
    cer = errors / total_chars if total_chars > 0 else 0
    return cer

def calculate_wer(predictions, ground_truth):
    """Calculate Word Error Rate (WER)"""
    errors = 0
    total_words = 0
    
    for pred, truth in zip(predictions, ground_truth):
        pred_words = str(pred).strip().split()
        truth_words = str(truth).strip().split()
        
        total_words += len(truth_words)
        if pred_words != truth_words:
            errors += max(len(pred_words), len(truth_words))
    
    wer = errors / total_words if total_words > 0 else 0
    return wer

# ============================================
# 🚀 INFERENCE
# ============================================
print(f"\n🧪 Testing on {len(test_dataset)} samples...\n")

predictions = []
ground_truths = []
correct_predictions = 0

with torch.no_grad():
    for idx, example in enumerate(tqdm(test_dataset, desc="Testing")):
        pixel_values = example["pixel_values"].unsqueeze(0).to(device)
        ground_truth = example["text"]
        
        # Generate prediction
        generated_ids = model.generate(pixel_values, max_length=64)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        predictions.append(predicted_text)
        ground_truths.append(ground_truth)
        
        # Count exact matches
        if predicted_text.strip().lower() == ground_truth.strip().lower():
            correct_predictions += 1

# ============================================
# 📈 RESULTS
# ============================================
accuracy = (correct_predictions / len(test_dataset)) * 100
cer = calculate_cer(predictions, ground_truths)
wer = calculate_wer(predictions, ground_truths)

print("\n" + "="*50)
print("📊 TEST RESULTS")
print("="*50)
print(f"Total samples tested: {len(test_dataset)}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Character Error Rate (CER): {cer:.4f}")
print(f"Word Error Rate (WER): {wer:.4f}")
print("="*50 + "\n")

# ============================================
# 🔍 SAMPLE PREDICTIONS
# ============================================
print("📝 Sample Predictions (first 10):\n")
for i in range(min(10, len(predictions))):
    print(f"Sample {i+1}:")
    print(f"  Ground Truth: {ground_truths[i]}")
    print(f"  Prediction:  {predictions[i]}")
    match = "✓" if predictions[i].strip().lower() == ground_truths[i].strip().lower() else "✗"
    print(f"  Result:      {match}\n")

# ============================================
# 💾 SAVE RESULTS
# ============================================
results_df = pd.DataFrame({
    "ground_truth": ground_truths,
    "prediction": predictions,
    "correct": [pred.strip().lower() == truth.strip().lower() for pred, truth in zip(predictions, ground_truths)]
})

results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.csv")
results_df.to_csv(results_file, index=False)
print(f"✅ Results saved to: {results_file}")
