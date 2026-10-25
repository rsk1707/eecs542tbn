"""ObjectRecord JSONL I/O (data/objects_{train,val}.jsonl). Owner: P2.

I/O is implemented; `build_object_records` is P2's job (COCO/LVIS -> records with filtering and
stratification fields, see docs/interfaces.md section 1 and open issues O1, O2, O10).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from promptcorrect.types import ObjectRecord


def save_objects(path: str | Path, records: Iterable[ObjectRecord]) -> int:
    """Write one JSON object per line. RLE `counts` must already be a str. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
            n += 1
    return n


def iter_objects(path: str | Path) -> Iterator[ObjectRecord]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield ObjectRecord(**json.loads(line))


def load_objects(path: str | Path) -> list[ObjectRecord]:
    return list(iter_objects(path))


def group_by_image(records: Iterable[ObjectRecord]) -> dict[int, list[ObjectRecord]]:
    out: dict[int, list[ObjectRecord]] = {}
    for r in records:
        out.setdefault(r.image_id, []).append(r)
    return out


def build_object_records(*args, **kwargs):
    """TODO(P2): COCO/LVIS annotations -> list[ObjectRecord]. Filter iscrowd and tiny objects once, here."""
    raise NotImplementedError
