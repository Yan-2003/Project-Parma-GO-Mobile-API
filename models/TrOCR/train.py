# train_trocr_improved_stable.py

import os
import pandas as pd
from PIL import Image
from datasets import Dataset
import torch
import re
import numpy as np

from torchvision import transforms

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
    s = re.sub(r"[^a-z0-9\s\-\.]", "", s)
    s = re.sub(r"\s+", " ", s)

    return s


# ============================================
# DATA AUGMENTATION (TRAIN ONLY)
# ============================================
augment = transforms.Compose([
    transforms.RandomRotation(2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])


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
# LOAD DATASET
# ============================================

def load_dataset(csv_path, image_dir):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"IMAGE", "MEDICINE_NAME", "GENERIC_NAME"}

    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required}")

    df["image_path"] = df["IMAGE"].apply(
        lambda x: os.path.join(image_dir, x)
    )

    df["text"] = (
        df["MEDICINE_NAME"].astype(str)
        + " "
        + df["GENERIC_NAME"].astype(str)
    ).apply(normalize_text)

    df = df[df["image_path"].apply(os.path.exists)]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No images found!")

    unique_labels = sorted(df["text"].unique())

    labels_path = os.path.join(
        os.path.dirname(csv_path),
        "unique_labels.npy"
    )

    np.save(labels_path, unique_labels)

    print(f"✅ Saved {len(unique_labels)} labels → {labels_path}")

    return Dataset.from_pandas(df[["image_path", "text"]]), unique_labels


# ============================================
# LOAD DATA
# ============================================
print("📂 Loading datasets...")

train_ds_raw, _ = load_dataset(TRAIN_CSV, TRAIN_IMG_DIR)
val_ds_raw, _ = load_dataset(VAL_CSV, VAL_IMG_DIR)


# ============================================
# MODEL
# ============================================
print("⚙️ Loading TrOCR Printed Model...")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-printed"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-printed"
)


model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = processor.tokenizer.vocab_size

model.config.max_length = 64
model.config.encoder.image_size = 384
model.config.num_beams = 4


# ============================================
# PREPROCESSING (SEPARATE TRAIN / VAL)
# ============================================

def preprocess_train(examples):

    images = []

    for path in examples["image_path"]:

        img = Image.open(path).convert("RGB")
        img = augment(img)  # only train
        images.append(img)

    pixel_values = processor(
        images,
        size=(384, 384),
        return_tensors="pt"
    ).pixel_values

    labels = processor.tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).input_ids

    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": list(pixel_values),
        "labels": list(labels),
    }


def preprocess_val(examples):

    images = []

    for path in examples["image_path"]:

        img = Image.open(path).convert("RGB")
        images.append(img)

    pixel_values = processor(
        images,
        size=(384, 384),
        return_tensors="pt"
    ).pixel_values

    labels = processor.tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).input_ids

    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": list(pixel_values),
        "labels": list(labels),
    }


print("🧪 Preprocessing...")

train_dataset = train_ds_raw.map(
    preprocess_train,
    batched=True,
    batch_size=16,
    remove_columns=["image_path", "text"],
)

val_dataset = val_ds_raw.map(
    preprocess_val,
    batched=True,
    batch_size=16,
    remove_columns=["image_path", "text"],
)


# ============================================
# TORCH FORMAT
# ============================================
train_dataset.set_format(
    type="torch",
    columns=["pixel_values", "labels"]
)

val_dataset.set_format(
    type="torch",
    columns=["pixel_values", "labels"]
)

print(f"✅ Train: {len(train_dataset)} | Val: {len(val_dataset)}")


# ============================================
# METRICS
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

            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )

        prev = cur

    return prev[lb]


unique_path = os.path.join(TRAIN_DIR, "unique_labels.npy")

unique_labels_set = set(
    np.load(unique_path)
) if os.path.exists(unique_path) else set()


def compute_metrics(eval_pred):

    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    pad_id = processor.tokenizer.pad_token_id

    labels = np.where(labels == -100, pad_id, labels)

    pred_txt = processor.batch_decode(
        preds,
        skip_special_tokens=True
    )

    label_txt = processor.batch_decode(
        labels,
        skip_special_tokens=True
    )

    pred_norm = [normalize_text(p) for p in pred_txt]
    label_norm = [normalize_text(l) for l in label_txt]

    accuracy = np.mean([
        p == l for p, l in zip(pred_norm, label_norm)
    ])

    cer = sum(
        levenshtein(p, l)
        for p, l in zip(pred_norm, label_norm)
    ) / max(sum(len(l) for l in label_norm), 1)

    return {
        "accuracy": accuracy,
        "cer": cer,
    }


# ============================================
# TRAINING CONFIG (MAC STABLE)
# ============================================
training_args = Seq2SeqTrainingArguments(

    output_dir="./results",

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    num_train_epochs=40,

    learning_rate=3e-5,

    warmup_steps=300,

    weight_decay=0.01,

    logging_steps=25,

    eval_strategy="epoch",
    save_strategy="epoch",

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    greater_is_better=True,

    predict_with_generate=True,

    generation_max_length=64,
    generation_num_beams=4,

    fp16=False,   # MPS stable

    seed=42,

    report_to="none",

    dataloader_num_workers=0,
)


# ============================================
# TRAINER
# ============================================
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

trainer = Seq2SeqTrainer(

    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    data_collator=lambda x: {
        "pixel_values": torch.stack(
            [i["pixel_values"] for i in x]
        ),
        "labels": torch.stack(
            [i["labels"] for i in x]
        ),
    },

    compute_metrics=compute_metrics,
)


# ============================================
# TRAIN
# ============================================
print("🚀 Starting training...")

trainer.train()

print("✅ Training finished!")


# ============================================
# SAVE MODEL
# ============================================
SAVE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model"
)

os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

model.generation_config.save_pretrained(SAVE_DIR)

print(f"💾 Model saved → {SAVE_DIR}")
