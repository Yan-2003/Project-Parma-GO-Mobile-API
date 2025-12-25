# train_trocr_fixed.py
import os
import pandas as pd
from PIL import Image
from datasets import Dataset
import torch
import re
import numpy as np

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ============================================
# TEXT NORMALIZATION
# ============================================
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


# ============================================
# PATHS
# ============================================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
VAL_DIR = os.path.join(BASE_DIR, "Validation")

TRAIN_CSV = os.path.join(TRAIN_DIR, "training_labels.csv")
VAL_CSV = os.path.join(VAL_DIR, "validation_labels.csv")
TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, "training_words")
VAL_IMG_DIR = os.path.join(VAL_DIR, "validation_words")


# ============================================
# LOAD DATASET + SAVE UNIQUE LABELS
# ============================================
def load_dataset(csv_path, image_dir):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"IMAGE", "MEDICINE_NAME", "GENERIC_NAME"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required}")

    df["image_path"] = df["IMAGE"].apply(lambda x: os.path.join(image_dir, x))
    df["text"] = (df["MEDICINE_NAME"].astype(str) + " " + df["GENERIC_NAME"].astype(str)).apply(normalize_text)

    # Keep only existing images
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No images found!")

    # Save unique labels for snapping
    unique_labels = sorted(df["text"].unique())
    labels_path = os.path.join(os.path.dirname(csv_path), "unique_labels.npy")
    np.save(labels_path, unique_labels)
    print(f"Saved {len(unique_labels)} unique labels → {labels_path}")

    return Dataset.from_pandas(df[["image_path", "text"]]), unique_labels


print("Loading datasets...")
train_ds_raw, train_unique_labels = load_dataset(TRAIN_CSV, TRAIN_IMG_DIR)
val_ds_raw, _ = load_dataset(VAL_CSV, VAL_IMG_DIR)


# ============================================
# MODEL & PROCESSOR
# ============================================
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Config fixes
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = processor.tokenizer.vocab_size
model.config.max_length = 64
model.config.early_stopping = False
model.config.num_beams = 4


# ============================================
# PREPROCESSING (BATCHED + SAFE)
# ============================================
def preprocess_function(examples):
    images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
    pixel_values = processor(images, return_tensors="pt").pixel_values  # [batch, 3, 384, 384]

    texts = examples["text"]
    labels = processor.tokenizer(
        texts,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": list(pixel_values),  # List of individual tensors [C, H, W]
        "labels": list(labels),
    }


print("Preprocessing...")
train_dataset = train_ds_raw.map(
    preprocess_function,
    batched=True,
    batch_size=16,
    remove_columns=["image_path", "text"],
)
val_dataset = val_ds_raw.map(
    preprocess_function,
    batched=True,
    batch_size=16,
    remove_columns=["image_path", "text"],
)

# Optional: Remove completely blank images (batched safe version)
def remove_blank_images(dataset):
    def is_not_blank(batch):
        # batch["pixel_values"] may contain tensors, numpy arrays or nested lists
        valid = []
        for pv in batch["pixel_values"]:
            try:
                t = pv if isinstance(pv, torch.Tensor) else torch.as_tensor(pv)
            except Exception:
                # fallback: convert via numpy then to tensor
                import numpy as _np
                t = torch.as_tensor(_np.array(pv))
            # any non-zero pixel indicates a non-blank image
            try:
                is_nonzero = bool(t.ne(0).any().item())
            except Exception:
                # final fallback: convert to cpu tensor and check
                is_nonzero = bool(torch.as_tensor(t).ne(0).any().cpu().item())
            valid.append(is_nonzero)
        return {"valid": valid}

    dataset = dataset.map(is_not_blank, batched=True, batch_size=32)
    dataset = dataset.filter(lambda x: x["valid"])
    return dataset.remove_columns(["valid"])

print("Removing blank images (if any)...")
train_dataset = remove_blank_images(train_dataset)
val_dataset = remove_blank_images(val_dataset)

print(f"Final sizes → Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Set torch format
train_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
val_dataset.set_format(type="torch", columns=["pixel_values", "labels"])


# ============================================
# LEVENSHTEIN + SNAPPING
# ============================================
def levenshtein(a, b):
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev = cur
    return prev[lb]

# Load unique labels for snapping
unique_labels_path = os.path.join(TRAIN_DIR, "unique_labels.npy")
unique_labels_set = set(np.load(unique_labels_path)) if os.path.exists(unique_labels_path) else set()


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    labels = np.where(labels == -100, pad_id, labels)

    pred_strs = processor.batch_decode(preds, skip_special_tokens=True)
    label_strs = processor.batch_decode(labels, skip_special_tokens=True)

    pred_norm = [normalize_text(p) for p in pred_strs]
    label_norm = [normalize_text(l) for l in label_strs]

    # Raw accuracy
    accuracy = sum(p == l for p, l in zip(pred_norm, label_norm)) / max(len(label_norm), 1)

    # Snapped accuracy (this will be high!)
    snapped = [
        min(unique_labels_set, key=lambda x: levenshtein(p, x)) if unique_labels_set and p.strip()
        else p for p in pred_norm
    ]
    snapped_accuracy = sum(s == l for s, l in zip(snapped, label_norm)) / max(len(label_norm), 1)

    cer = sum(levenshtein(p, l) for p, l in zip(pred_norm, label_norm)) / max(sum(len(l) for l in label_norm), 1)
    wer = sum(levenshtein(p.split(), l.split()) for p, l in zip(pred_norm, label_norm)) / max(sum(len(l.split()) for l in label_norm), 1)

    return {
        "accuracy": accuracy,
        "snapped_accuracy": snapped_accuracy,
        "cer": cer,
        "wer": wer,
    }


# ============================================
# TRAINING
# ============================================
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    learning_rate=1e-5,
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="snapped_accuracy",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=64,
    generation_num_beams=4,
    fp16=False,  # Disabled for Apple Silicon stability
    seed=42,
    report_to="none",
    dataloader_num_workers=0,  # Important on macOS/Windows
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda x: {
        "pixel_values": torch.stack([i["pixel_values"] for i in x]),
        "labels": torch.stack([i["labels"] for i in x]),
    },
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()
print("Training finished!")


# ============================================
# SAVE MODEL
# ============================================
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
model.generation_config.save_pretrained(SAVE_DIR)
print(f"Model saved → {SAVE_DIR}")


# ============================================
# INFERENCE WITH STRICT LABEL CONSTRAINT
# ============================================
def predict(image_path: str) -> str:
    """Predict text from image, constrained to valid training labels only."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(model.device)

    # Generate multiple candidates
    generated = model.generate(
        pixel_values,
        max_length=64,
        num_beams=4,
        num_return_sequences=4,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True,
    )
    
    # Try each beam candidate and return first exact match to training labels
    for beam_idx in range(min(4, len(generated.sequences))):
        text = processor.decode(generated.sequences[beam_idx], skip_special_tokens=True)
        norm = normalize_text(text)
        if norm in unique_labels_set:
            return norm
    
    # If no exact match, snap to closest training label
    text = processor.decode(generated.sequences[0], skip_special_tokens=True)
    norm = normalize_text(text)
    return min(unique_labels_set, key=lambda x: levenshtein(norm, x)) if unique_labels_set else norm


# Example:
# print(predict("data/Validation/validation_words/your_image.jpg"))