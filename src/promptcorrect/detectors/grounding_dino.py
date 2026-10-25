"""Grounding DINO demo fallback, using the Hugging Face Transformers API."""

from typing import Any


class GroundingDinoDetector:
    """Return original-image xyxy boxes and confidence for one GT phrase."""

    def __init__(
        self, model_id: str = "IDEA-Research/grounding-dino-tiny", *,
        device: str = "cpu", threshold: float = 0.35, text_threshold: float = 0.25,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if not 0 <= threshold <= 1 or not 0 <= text_threshold <= 1:
            raise ValueError("Detection thresholds must be in [0, 1]")
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, revision=revision
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.torch = torch

    def detect(self, image: Any, phrase: str) -> list[tuple[list[float], float]]:
        """Run one phrase at a time so every output has unambiguous provenance."""
        phrase = phrase.strip().replace("_", " ").rstrip(".")
        if not phrase:
            raise ValueError("Detection phrase must not be empty")
        image = image.convert("RGB")
        inputs = self.processor(images=image, text=phrase + ".", return_tensors="pt")
        inputs = inputs.to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.threshold,
            text_threshold=self.text_threshold, target_sizes=[(image.height, image.width)],
        )[0]
        return [(list(map(float, box)), float(score)) for box, score in
                zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(), strict=True)]
