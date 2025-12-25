# test_trocr_fixed.py
import os
import re
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

# ============================================
# TEXT NORMALIZATION (MUST MATCH TRAIN)
# ============================================
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

# ============================================
# LEVENSHTEIN DISTANCE
# ============================================
def levenshtein(a, b):
    if a == b:
        return 0
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

# ============================================
# PATHS
# ============================================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TEST_DIR = os.path.join(BASE_DIR, "Testing")
TEST_CSV = os.path.join(TEST_DIR, "testing_labels.csv")
TEST_IMG_DIR = os.path.join(TEST_DIR, "testing_words")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
UNIQUE_LABELS_PATH = os.path.join(TRAIN_DIR, "unique_labels.npy")

# ============================================
# LOAD UNIQUE LABELS (FOR SNAPPING)
# ============================================
if not os.path.exists(UNIQUE_LABELS_PATH):
    raise FileNotFoundError("❌ unique_labels.npy not found. Retrain the model.")

unique_labels = np.load(UNIQUE_LABELS_PATH)
unique_labels_set = set(unique_labels)

print(f"✅ Loaded {len(unique_labels)} unique labels")

# ============================================
# LOAD DATASET
# ============================================
def load_dataset(csv_path, image_dir):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV missing → {csv_path}")

    df = pd.read_csv(csv_path)
    df["image_path"] = df["IMAGE"].apply(lambda x: os.path.join(image_dir, x))
    df["text"] = (
        df["MEDICINE_NAME"].astype(str) + " " + df["GENERIC_NAME"].astype(str)
    ).apply(normalize_text)

    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return Dataset.from_pandas(df[["image_path", "text"]])

print("📂 Loading test dataset...")
test_dataset = load_dataset(TEST_CSV, TEST_IMG_DIR)
print(f"✅ Test samples: {len(test_dataset)}")

# ============================================
# LOAD MODEL
# ============================================
print("⚙️ Loading model...")
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

print(f"📱 Using device: {device}")

# ============================================
# INFERENCE + EVALUATION
# ============================================
predictions = []
ground_truths = []
correct = 0
snapped_correct = 0

def constrain_to_valid_labels(pred_text, valid_labels):
    """Force prediction to be one of the valid training labels.
    If exact match found, return it. Otherwise return closest label by edit distance."""
    pred_norm = normalize_text(pred_text)
    
    # Check for exact match first
    if pred_norm in valid_labels:
        return pred_norm
    
    # Otherwise snap to closest label
    if valid_labels:
        return min(valid_labels, key=lambda x: levenshtein(pred_norm, x))
    return pred_norm

with torch.no_grad():
    for example in tqdm(test_dataset, desc="Testing"):
        image = Image.open(example["image_path"]).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # Generate multiple beam candidates
        generated = model.generate(
            pixel_values,
            max_length=64,
            num_beams=4,
            num_return_sequences=4,  # Get top 4 candidates
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Try each beam candidate in order and pick the first one that matches a valid label
        generated_ids = generated.sequences
        best_pred = None
        
        for beam_idx in range(min(4, len(generated_ids))):
            raw_pred = processor.decode(generated_ids[beam_idx], skip_special_tokens=True)
            pred_norm = normalize_text(raw_pred)
            
            # If this candidate is in valid labels, use it
            if pred_norm in unique_labels_set:
                best_pred = pred_norm
                break
        
        # If no exact match found in any beam, use closest label
        if best_pred is None:
            raw_pred = processor.decode(generated_ids[0], skip_special_tokens=True)
            best_pred = constrain_to_valid_labels(raw_pred, unique_labels_set)

        truth = example["text"]

        predictions.append(best_pred)
        ground_truths.append(truth)

        # Raw accuracy (before snapping)
        raw_matches = any(
            normalize_text(processor.decode(generated_ids[i], skip_special_tokens=True)) == truth
            for i in range(len(generated_ids))
        )
        if raw_matches:
            correct += 1
        
        # Exact match after constraint
        if best_pred == truth:
            snapped_correct += 1

# ============================================
# METRICS
# ============================================
def cer(preds, truths):
    total_err, total_chars = 0, 0
    for p, t in zip(preds, truths):
        total_err += levenshtein(p, t)
        total_chars += len(t)
    return total_err / max(total_chars, 1)

def wer(preds, truths):
    total_err, total_words = 0, 0
    for p, t in zip(preds, truths):
        total_err += levenshtein(p.split(), t.split())
        total_words += len(t.split())
    return total_err / max(total_words, 1)

accuracy = correct / len(test_dataset) * 100
constrained_accuracy = snapped_correct / len(test_dataset) * 100
cer_score = cer(predictions, ground_truths)
wer_score = wer(predictions, ground_truths)

# ============================================
# RESULTS
# ============================================
print("\n" + "=" * 60)
print("📊 FINAL TEST RESULTS (CONSTRAINED TO TRAINING LABELS)")
print("=" * 60)
print(f"Samples tested:              {len(test_dataset)}")
print(f"Raw accuracy (any beam):     {accuracy:.2f}%")
print(f"Constrained accuracy:        {constrained_accuracy:.2f}%")
print(f"  (Only valid training labels allowed)")
print(f"Character Error Rate:        {cer_score:.4f}")
print(f"Word Error Rate:             {wer_score:.4f}")
print(f"Valid unique labels:         {len(unique_labels_set)}")
print("=" * 60)

# ============================================
# SAMPLE OUTPUTS
# ============================================
print("\n📝 Sample predictions:\n")
for i in range(min(10, len(predictions))):
    print(f"[{i+1}] GT: {ground_truths[i]}")
    print(f"     PR: {predictions[i]}")
    print("")

# ============================================
# SAVE RESULTS
# ============================================
results_df = pd.DataFrame({
    "ground_truth": ground_truths,
    "prediction": predictions,
    "correct": [p == t for p, t in zip(predictions, ground_truths)]
})

results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.csv")
results_df.to_csv(results_path, index=False)
print(f"✅ Results saved → {results_path}")