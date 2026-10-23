"""Owner: P1 (Srikrishnan). Segmenter interface, backends, and the factory."""

from promptcorrect.segmenters.base import Segmenter


def make_segmenter(backend: str, **kwargs) -> Segmenter:
    """Build a segmenter by name: "dummy" | "sam3" | "sam21". Backends import lazily (heavy deps)."""
    if backend == "dummy":
        from promptcorrect.segmenters.dummy import DummySegmenter

        return DummySegmenter(**kwargs)
    if backend == "sam3":
        from promptcorrect.segmenters.sam3 import Sam3Segmenter

        return Sam3Segmenter(**kwargs)
    if backend == "sam21":
        from promptcorrect.segmenters.sam21 import Sam21Segmenter

        return Sam21Segmenter(**kwargs)
    raise ValueError(f"Unknown backend {backend!r} (expected dummy | sam3 | sam21)")


__all__ = ["Segmenter", "make_segmenter"]
