"""Lightweight TrOCR helper module.

Provides a small `TrOCRModel` class to load a HuggingFace TrOCR processor+model
and run inference on PIL images or image file paths. Intended for quick use
inside other scripts (e.g. `scripts/run_trocr_infer.py`) or from a REPL.

Example:
    from models.TrOCR_V2.model import TrOCRModel
    trocr = TrOCRModel()
    trocr.load("microsoft/trocr-base-handwritten")
    text = trocr.predict_from_path("test.jpg")
    print(text)

"""

from typing import List, Union, Optional
import os
from PIL import Image
import torch

from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TrOCRModel:
	"""Wrapper around TrOCR processor + VisionEncoderDecoderModel.

	Methods:
	- load(model_name_or_path, device=None)
	- predict(images)
	- predict_from_path(path)
	"""

	def __init__(self):
		self.processor: Optional[TrOCRProcessor] = None
		self.model: Optional[VisionEncoderDecoderModel] = None
		self.device: str = "cpu"

	def load(self, model_name_or_path: str = "microsoft/trocr-base-handwritten", device: Optional[str] = None):
		"""Load processor and model from HF model hub or local path.

		Args:
			model_name_or_path: HF model identifier or local path.
			device: Optional device string ("cpu" or "cuda"). Auto-detect if None.
		"""
		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"

		self.device = device

		self.processor = TrOCRProcessor.from_pretrained(model_name_or_path)
		self.model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)

		# ensure tokens are set
		if getattr(self.model.config, "decoder_start_token_id", None) is None:
			self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
		if getattr(self.model.config, "eos_token_id", None) is None:
			self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
		if getattr(self.model.config, "pad_token_id", None) is None:
			self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

		self.model.to(self.device)
		self.model.eval()

	def _ensure_loaded(self):
		if self.processor is None or self.model is None:
			raise RuntimeError("Model not loaded. Call `load(...)` first.")

	def predict(self, images: List[Image.Image], max_length: int = 32, num_beams: int = 4) -> List[str]:
		"""Run inference on a list of PIL images and return decoded strings."""
		self._ensure_loaded()

		# processor can accept list of PIL images
		encoding = self.processor(images=images, return_tensors="pt")

		pixel_values = encoding.pixel_values.to(self.device)

		gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

		with torch.no_grad():
			generated_ids = self.model.generate(pixel_values, **gen_kwargs)

		preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
		return [p.strip() for p in preds]

	def predict_from_path(self, image_path: Union[str, os.PathLike], max_length: int = 32, num_beams: int = 4) -> str:
		"""Load a single image from path and run inference returning the text."""
		img = Image.open(str(image_path)).convert("RGB")
		return self.predict([img], max_length=max_length, num_beams=num_beams)[0]


def recognize_image(path: str, model: str = "microsoft/trocr-base-handwritten", device: Optional[str] = None) -> str:
	"""Convenience function: load model, run on single image path, return text."""
	tr = TrOCRModel()
	tr.load(model, device=device)
	return tr.predict_from_path(path)


if __name__ == "__main__":
	# Simple CLI for quick testing
	import argparse
	import json
	import sys
	import warnings
	
	warnings.filterwarnings("ignore")

	parser = argparse.ArgumentParser(description="Run TrOCR on an input image")
	parser.add_argument("image", help="Path to input image")
	parser.add_argument("--model", default="microsoft/trocr-base-handwritten", help="Model name or local path")
	parser.add_argument("--device", default=None, help="Device to use (cpu or cuda). Auto-detected if omitted")
	parser.add_argument("--beams", type=int, default=4, help="Number of beams for generation")
	parser.add_argument("--max-len", type=int, default=32, help="Maximum generation length")

	args = parser.parse_args()

	try:
		tr = TrOCRModel()
		tr.load(args.model, device=args.device)
		text = tr.predict_from_path(args.image, max_length=args.max_len, num_beams=args.beams)
		print(json.dumps({"text": text.strip()}))
	except Exception as e:
		print(json.dumps({"error": str(e)}))
		sys.exit(1)

