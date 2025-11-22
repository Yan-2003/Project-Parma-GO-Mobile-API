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

    return Dataset.from_pandas(df[["image_path", "text"]])


print("📂 Loading datasets...")
train_dataset = load_dataset(TRAIN_CSV, TRAIN_IMG_DIR)
val_dataset = load_dataset(VAL_CSV, VAL_IMG_DIR)

# ============================================
# ⚙️ LOAD PROCESSOR + MODEL
# ============================================
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Fix token IDs
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

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
    labels = processor.tokenizer(
        example["text"],
        max_length=64,
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
from difflib import SequenceMatcher

def compute_metrics(pred):
    """Compute accuracy metrics during evaluation."""
    predictions = pred.predictions
    labels = pred.label_ids
    
    # Replace -100 (ignore tokens) with pad token
    pad_token_id = processor.tokenizer.pad_token_id
    predictions[predictions == -100] = pad_token_id
    labels[labels == -100] = pad_token_id
    
    # Decode predictions and labels
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate exact match accuracy
    exact_matches = sum(1 for pred, label in zip(decoded_preds, decoded_labels) 
                       if pred.strip().lower() == label.strip().lower())
    accuracy = exact_matches / len(decoded_preds) if decoded_preds else 0
    
    return {"accuracy": accuracy}

# ============================================
# 🧩 TRAINING ARGUMENTS (FULLY COMPATIBLE)
# ============================================
common_args = dict(
    output_dir="./results",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_strategy="epoch",
    predict_with_generate=True,
    report_to="none",
    load_best_model_at_end=True,
)

try:
    # New transformers versions
    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy="epoch",
        **common_args
    )
except TypeError:
    # Older versions
    training_args = Seq2SeqTrainingArguments(
        eval_strategy="epoch",
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