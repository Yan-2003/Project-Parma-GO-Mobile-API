"""run_trocr_infer.py
Simple CLI to run TrOCR (handwritten) on an input image and print recognized text.

Usage:
  python scripts/run_trocr_infer.py --image path/to/image.jpg
  python scripts/run_trocr_infer.py --image img1.jpg img2.png --model microsoft/trocr-base-handwritten

Options:
  --model   Model name or path (default: microsoft/trocr-base-handwritten)
  --device  Device to run on, e.g. cpu or cuda (auto-detected by default)
  --out     Optional output file to append results as CSV

The script uses Hugging Face Transformers `TrOCRProcessor` + `VisionEncoderDecoderModel`.
"""

import argparse
import os
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import List
import csv


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def run_inference_on_images(
    image_paths: List[str], model_name_or_path: str, device: str = None
) -> List[str]:
    # device auto-detect
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading processor and model: {model_name_or_path} on {device}...")
    processor = TrOCRProcessor.from_pretrained(model_name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)

    model.to(device)
    model.eval()

    # ensure model start/eos/pad are set (usually they are for trocr pretrained) but safe-guard
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = processor.tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    results = []

    # batch processing (small batches) - we will load all images then process in a small batch
    images = [load_image(p) for p in image_paths]

    # processor can accept list of PIL images
    encoding = processor(images=images, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(device)

    # generation parameters — adjust for your needs
    gen_kwargs = dict(max_length=128, num_beams=4)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, **gen_kwargs)

    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for img_path, pred in zip(image_paths, preds):
        results.append(pred)
        print(f"\nImage: {img_path}\nRecognized: {pred}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run TrOCR handwriting recognition on image(s)")
    parser.add_argument("--image", "-i", nargs="+", required=True, help="Path(s) to image file(s)")
    parser.add_argument("--model", "-m", default="microsoft/trocr-base-handwritten", help="Model name or local path")
    parser.add_argument("--device", "-d", default=None, help="Device to run on (cpu or cuda). Auto-detect if omitted")
    parser.add_argument("--out", "-o", default=None, help="Optional CSV output file to append results")

    args = parser.parse_args()

    image_paths = args.image
    model_name = args.model
    device = args.device

    try:
        results = run_inference_on_images(image_paths, model_name, device)
    except Exception as e:
        print(f"Error during inference: {e}")
        raise

    # optionally append to CSV
    if args.out:
        out_file = args.out
        header = ["image", "recognized_text"]
        file_exists = os.path.exists(out_file)
        with open(out_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            for img, txt in zip(image_paths, results):
                writer.writerow([img, txt])
        print(f"Results appended to: {out_file}")


if __name__ == "__main__":
    main()
