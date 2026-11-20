"""LVIS records and explicit COCO/LVIS novelty review. Owner: P2.

Synset equality supplies review candidates, not a claim that every other LVIS
category is unseen: COCO's broad categories can include narrower LVIS classes.
"""

from collections.abc import Iterable, Iterator
from dataclasses import replace

from promptcorrect.types import ObjectRecord

from .records import build_coco_records


def _categories_by_id(categories: list[dict]) -> dict[int, dict]:
    indexed = {c["id"]: c for c in categories}
    if len(indexed) != len(categories) or not indexed:
        raise ValueError("Categories must have unique IDs and must not be empty")
    return indexed


def draft_category_review(coco_categories: list[dict], lvis_categories: list[dict],
                          coco_synsets: dict) -> dict:
    """Draft decisions from the official LVIS coco_to_synset.json vocabulary.

    Every row starts unreviewed; unmatched synsets have no novelty decision.
    The draft binds each decision to the category's ID, name and synset.
    """
    coco = _categories_by_id(coco_categories)
    lvis = _categories_by_id(lvis_categories)
    supplied = {v["coco_cat_id"]: (name, v["synset"]) for name, v in coco_synsets.items()}
    if len(supplied) != len(coco_synsets) or set(supplied) != set(coco):
        raise ValueError("Synset mapping must cover exactly the COCO category IDs")
    if any(supplied[i][0] != c["name"] for i, c in coco.items()):
        raise ValueError("Synset mapping names disagree with COCO categories")
    by_synset = {}
    for ident, (_, synset) in supplied.items():
        by_synset.setdefault(synset, []).append(ident)
    return {"schema_version": 1, "categories": [
        {"lvis_category_id": i, "name": c["name"], "synset": c["synset"],
         "coco_category_ids": sorted(by_synset.get(c["synset"], [])),
         "novelty": "seen" if c["synset"] in by_synset else None,
         "reviewed": False, "review_notes": ""}
        for i, c in sorted(lvis.items())
    ]}


def validate_category_review(review: dict, coco_categories: list[dict],
                             lvis_categories: list[dict]) -> dict[int, str]:
    """Require a documented decision for every category, including absent ones."""
    coco = _categories_by_id(coco_categories)
    lvis = _categories_by_id(lvis_categories)
    if review.get("schema_version") != 1:
        raise ValueError("Unsupported category review schema_version")
    rows = review["categories"]
    decisions = {r["lvis_category_id"]: r for r in rows}
    if len(decisions) != len(rows) or set(decisions) != set(lvis):
        raise ValueError("Review must cover every LVIS category exactly once")
    for ident, row in decisions.items():
        category = lvis[ident]
        if any(row[k] != category[k] for k in ("name", "synset")):
            raise ValueError(f"Stale category review for LVIS {ident}")
        notes = row.get("review_notes")
        if row.get("reviewed") is not True or not isinstance(notes, str) or not notes.strip():
            raise ValueError(f"LVIS {ident} needs a reviewed decision and review_notes")
        novelty, matches = row.get("novelty"), row.get("coco_category_ids")
        if novelty not in ("seen", "unseen") or not isinstance(matches, list):
            raise ValueError(f"Invalid novelty decision for LVIS {ident}")
        if len(set(matches)) != len(matches) or set(matches) - coco.keys():
            raise ValueError(f"Invalid COCO category IDs for LVIS {ident}")
        if (novelty == "seen") != bool(matches):
            raise ValueError(f"Seen categories require COCO matches; unseen require none: {ident}")
    return {i: r["novelty"] for i, r in decisions.items()}


def build_lvis_records(annotations: dict, novelty_by_category: dict[int, str],
                       image_ids: Iterable[int] | None = None) -> Iterator[ObjectRecord]:
    """Reuse P2's mask/filter/stratification implementation with LVIS metadata.

    LVIS image IDs are preserved (shared COCO IDs); annotation IDs use `lvis:`.
    Phrases use the first synonym, with underscores replaced by spaces.
    """
    categories = _categories_by_id(annotations["categories"])
    if set(novelty_by_category) != set(categories) or any(
        n not in ("seen", "unseen") for n in novelty_by_category.values()
    ):
        raise ValueError("Explicit seen/unseen labels required for every LVIS category")
    adapted = []
    for category in categories.values():
        synonyms = category.get("synonyms")
        if not synonyms or not isinstance(synonyms[0], str) or not synonyms[0].strip():
            raise ValueError(f"Missing first synonym for LVIS {category['id']}")
        adapted.append({**category, "name": synonyms[0]})
    shared = {**annotations, "categories": adapted}
    for record in build_coco_records(shared, image_ids):
        yield replace(record, obj_id="lvis:" + record.obj_id.removeprefix("coco:"),
                      source="lvis", novelty=novelty_by_category[record.category_id])
