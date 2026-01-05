#!/usr/bin/env python3
"""Pharma_TrOCRV2.py

Replacement for `Pharma_TrOCR.py` that uses the helper module at
`models/TrOCR-V2/model.py` (loads it by path so the hyphen in the folder
name does not require a Python package import).

Usage:
    python scripts/Pharma_TrOCRV2.py <image_path> [--model-dir <model_dir>]

Outputs JSON to stdout: {"text": "..."} or {"error": "..."}
"""

import sys
import os
import json
import warnings
from PIL import Image
import torch
import importlib.util
from typing import Optional

warnings.filterwarnings("ignore")

# Path to the helper module file (models/TrOCR-V2/model.py)
MODULE_FILE = os.path.join(os.path.dirname(__file__), "../models/TrOCR-V2/model.py")
MODULE_FILE = os.path.normpath(MODULE_FILE)

if not os.path.exists(MODULE_FILE):
    # Try alternate casing (in case of capitalized folder)
    alt = os.path.join(os.path.dirname(__file__), "../Models/TrOCR-V2/model.py")
    alt = os.path.normpath(alt)
    if os.path.exists(alt):
        MODULE_FILE = alt

# load the module from file path to avoid import name issues
def load_trocr_module(path: str):
    spec = importlib.util.spec_from_file_location("trocr_v2_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    # Basic CLI parsing (kept simple for compatibility)
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python Pharma_TrOCRV2.py <image_path> --model-dir <model_dir>"}))
        sys.exit(1)

    image_path = sys.argv[1]
    model_dir = None
    if "--model-dir" in sys.argv:
        try:
            idx = sys.argv.index("--model-dir")
            model_dir = sys.argv[idx + 1]
        except Exception:
            model_dir = None

    if not os.path.exists(image_path):
        print(json.dumps({"error": f"File not found: {image_path}"}))
        sys.exit(1)

    # Model directory is required
    if not model_dir:
        print(json.dumps({"error": "Model directory required. Usage: python Pharma_TrOCRV2.py <image_path> --model-dir <model_dir>"}))
        sys.exit(1)

    model_dir = os.path.normpath(model_dir)
    if not os.path.exists(model_dir):
        print(json.dumps({"error": f"Model directory not found: {model_dir}"}))
        sys.exit(1)

    # device selection: prefer cuda, fallback to mps (apple silicon), then cpu
    device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"

    # Load helper module
    try:
        module = load_trocr_module(MODULE_FILE)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load TrOCR helper module: {e}"}))
        sys.exit(1)

    # Ensure TrOCRModel exists
    if not hasattr(module, "TrOCRModel"):
        print(json.dumps({"error": "TrOCR helper module does not define TrOCRModel"}))
        sys.exit(1)

    try:
        trocr = module.TrOCRModel()
        trocr.load(model_dir, device=device)

        # run prediction
        text = trocr.predict_from_path(image_path)
        print(json.dumps({"text": text.strip()}))

    except Exception as e:
        # return the error as JSON so API callers can parse it
        print(json.dumps({"error": str(e)}))
        sys.exit(2)


if __name__ == "__main__":
    main()
