"""Single implementation of prompt severity. Owner: P4."""

from __future__ import annotations

from promptcorrect.types import ObjectRecord, Prompt


def describe(prompt: Prompt, obj: ObjectRecord, neighbors: list[ObjectRecord]) -> dict:
    """TODO(P4): -> {box_iou, coverage, contamination, delta}.

    box_iou: IoU of prompt box with GT box. coverage: fraction of GT mask inside the box.
    contamination: fraction of the box covered by `neighbors`' masks (other objects in the image).
    delta: (dcx/w, dcy/h, log s_w, log s_h) relative to the GT box.
    Box-based fields are NaN when prompt.box is None (docs/interfaces.md, issue O3).
    """
    raise NotImplementedError
