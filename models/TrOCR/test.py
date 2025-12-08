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
import re

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
        max_length=32,
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
# (Replaced earlier naive CER implementation with Levenshtein-based functions below)

def _levenshtein_seq(a, b):
    """Compute Levenshtein distance between two sequences (strings or lists)."""
    # convert to lists if strings
    if isinstance(a, str) and isinstance(b, str):
        a = list(a)
        b = list(b)

    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]

def normalize_text(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    # keep only alphanumeric and spaces
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def calculate_cer(predictions, ground_truth):
    """Character Error Rate using Levenshtein distance over characters."""
    total_errors = 0
    total_chars = 0
    for pred, truth in zip(predictions, ground_truth):
        p = normalize_text(pred)
        t = normalize_text(truth)
        total_errors += _levenshtein_seq(p, t)
        total_chars += len(t)
    return (total_errors / total_chars) if total_chars > 0 else 0

def calculate_wer(predictions, ground_truth):
    """Word Error Rate using Levenshtein distance over words."""
    total_errors = 0
    total_words = 0
    for pred, truth in zip(predictions, ground_truth):
        p_words = normalize_text(pred).split()
        t_words = normalize_text(truth).split()
        total_errors += _levenshtein_seq(p_words, t_words)
        total_words += len(t_words)
    return (total_errors / total_words) if total_words > 0 else 0


# ============================================
# 🚀 INFERENCE
# ============================================
print(f"\n🧪 Testing on {len(test_dataset)} samples...\n")

predictions = []
ground_truths = []
correct_predictions = 0
medicine_prefix_matches = 0
fuzzy_correct = 0

# load raw labels to evaluate medicine-only matches
raw_df = pd.read_csv(TEST_CSV)
med_list = raw_df["MEDICINE_NAME"].astype(str).tolist()

with torch.no_grad():
    for idx, example in enumerate(tqdm(test_dataset, desc="Testing")):
        pixel_values = example["pixel_values"].unsqueeze(0).to(device)
        ground_truth = example["text"]

        # Generate prediction (use beam search to improve outputs)
        generated_ids = model.generate(pixel_values, max_length=32, num_beams=4, early_stopping=True)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        predictions.append(predicted_text)
        ground_truths.append(ground_truth)

        # normalized exact-match
        if normalize_text(predicted_text) == normalize_text(ground_truth):
            correct_predictions += 1

        # fuzzy correctness: accept small CER as correct (helps diagnose near-misses)
        p_norm = normalize_text(predicted_text)
        t_norm = normalize_text(ground_truth)
        cer_sample = _levenshtein_seq(p_norm, t_norm) / (len(t_norm) if len(t_norm) > 0 else 1)
        if cer_sample <= 0.20:
            fuzzy_correct += 1

        # medicine-name prefix match (useful because labels are "MEDICINE GENERIC")
        med = med_list[idx] if idx < len(med_list) else ""
        med_words = normalize_text(med).split()
        pred_words = normalize_text(predicted_text).split()
        if med_words and pred_words[: len(med_words)] == med_words:
            medicine_prefix_matches += 1

# ============================================
# 📈 RESULTS
# ============================================
accuracy = (correct_predictions / len(test_dataset)) * 100
fuzzy_accuracy = (fuzzy_correct / len(test_dataset)) * 100
cer = calculate_cer(predictions, ground_truths)
wer = calculate_wer(predictions, ground_truths)

print("\n" + "="*50)
print("📊 TEST RESULTS")
print("="*50)
print(f"Total samples tested: {len(test_dataset)}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Fuzzy accuracy (CER<=0.20): {fuzzy_accuracy:.2f}%")
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
