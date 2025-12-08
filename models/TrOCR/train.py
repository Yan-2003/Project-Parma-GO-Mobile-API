import os
import pandas as pd
from PIL import Image
from datasets import Dataset
import torch
import re

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# simple normalization used for labels and evaluation
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

# ============================================
# 📂 CORRECT PATH SETUP
# ============================================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TRAIN_DIR = os.path.join(BASE_DIR, "Training")
VAL_DIR = os.path.join(BASE_DIR, "Validation")

TRAIN_CSV = os.path.join(TRAIN_DIR, "training_labels.csv")
VAL_CSV = os.path.join(VAL_DIR, "validation_labels.csv")

TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, "training_words")
VAL_IMG_DIR = os.path.join(VAL_DIR, "validation_words")

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
    # normalize labels so model sees consistent text format
    df["text"] = df["text"].apply(normalize_text)

    return Dataset.from_pandas(df[["image_path", "text"]])


print("📂 Loading datasets...")
train_dataset = load_dataset(TRAIN_CSV, TRAIN_IMG_DIR)
val_dataset = load_dataset(VAL_CSV, VAL_IMG_DIR)

# ============================================
# ⚙️ LOAD PROCESSOR + MODEL
# ============================================
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Fix token IDs for proper decoding
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = processor.tokenizer.vocab_size
model.config.max_length = 32  # Enforce strict max generation length
# Ensure generation config is valid on save: don't enable early_stopping
# unless beam search (num_beams>1) will be used by default.
model.config.early_stopping = False
model.config.num_beams = 4

# ============================================
# 🧠 PREPROCESS FUNCTION
# ============================================
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    # produce a single-sample tensor and remove the leading batch dim
    pixel_values = processor(
        image,
        return_tensors="pt"
    ).pixel_values.squeeze(0)

    # tokenize the label text and replace pad token ids with -100 so loss ignores padding
    # lowercase targets so model sees consistent casing (metrics also lowercase)
    targets = str(example.get("text", "")).lower()
    labels = processor.tokenizer(
        targets,
        max_length=32,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).input_ids[0]

    # replace padding token id with -100 (ignore index for the loss)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    labels = labels.clone()
    labels[labels == pad_token_id] = -100

    return {"pixel_values": pixel_values, "labels": labels}

print("⚙️ Preprocessing data...")
train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

# Ensure datasets return PyTorch tensors (keeps pixel_values and labels as tensors)
train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")

# ============================================
# 📊 COMPUTE METRICS FUNCTION
# ============================================
import re
import numpy as np


def _levenshtein(a: str, b: str) -> int:
    # simple levenshtein for CER/WER if needed
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


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def compute_metrics(pred):
    """Compute accuracy, CER and WER during evaluation.

    Handles cases where `pred.predictions` can be a tuple (generated_ids, ...)
    and ensures proper decoding before comparison.
    """
    predictions = pred.predictions
    label_ids = pred.label_ids

    # If predictions is a tuple (generated_ids, scores), take first element
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.array(predictions)
    label_ids = np.array(label_ids)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    # Replace -100 in labels with pad token id for decoding
    label_ids = np.where(label_ids == -100, pad_token_id, label_ids)

    # Also ensure predictions don't contain -100
    predictions = np.where(predictions == -100, pad_token_id, predictions)

    # Decode
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize for comparison (strip whitespace and lowercase)
    norm_preds = [p.strip().lower() for p in decoded_preds]
    norm_labels = [l.strip().lower() for l in decoded_labels]

    # Exact-match accuracy
    exact = sum(1 for p, l in zip(norm_preds, norm_labels) if p == l)
    total = len(norm_preds) if len(norm_preds) > 0 else 1
    accuracy = exact / total

    # CER and WER
    total_cer = 0
    total_chars = 0
    total_wer = 0
    total_words = 0
    for p, l in zip(norm_preds, norm_labels):
        total_cer += _levenshtein(p, l)
        total_chars += len(l) if len(l) > 0 else 1
        p_words = p.split()
        l_words = l.split()
        total_wer += _levenshtein(p_words, l_words)
        total_words += len(l_words) if len(l_words) > 0 else 1

    cer = (total_cer / total_chars) if total_chars > 0 else 0
    wer = (total_wer / total_words) if total_words > 0 else 0

    return {"accuracy": accuracy, "cer": cer, "wer": wer}

# ============================================
# 🧩 TRAINING ARGUMENTS (FULLY COMPATIBLE)
# ============================================
common_args = dict(
    output_dir="./results",
    save_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=15,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    save_steps=780,
    eval_steps=780,
    predict_with_generate=True,
    generation_max_length=32,
    generation_num_beams=4,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

try:
    # New transformers versions (evaluation_strategy is the new name)
    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy="steps",
        **common_args
    )
except TypeError:
    # Older versions (eval_strategy is the old name)
    training_args = Seq2SeqTrainingArguments(
        eval_strategy="steps",
        **common_args
    )

# ============================================
# 🚀 TRAINER
# ============================================
def collate_fn(features):
    """Collate batch of examples."""
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

print(f"🚀 Training started: {len(train_dataset)} samples")
trainer.train()
print("🎉 Training complete!")

# ============================================
# 💾 SAVE FINAL MODEL
# ============================================
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

print(f"✅ Final model saved at: {SAVE_DIR}")