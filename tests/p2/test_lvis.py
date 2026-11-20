from copy import deepcopy

import pytest

from promptcorrect.data.lvis import (
    build_lvis_records, draft_category_review, validate_category_review,
)
from promptcorrect.data.records import build_coco_records, decode_rle
from test_records import coco_fixture


def lvis_fixture():
    data = coco_fixture()
    data["categories"] = [
        {"id": 1, "name": "different_name", "synset": "test.n.01",
         "synonyms": ["first_synonym", "alternate"]},
        {"id": 2, "name": "novel", "synset": "novel.n.01", "synonyms": ["novel"]},
    ]
    # LVIS annotations need not carry COCO's iscrowd field.
    data["annotations"] = [a for a in data["annotations"] if not a.pop("iscrowd")]
    return data


def reviewed_fixture():
    coco, lvis = coco_fixture(), lvis_fixture()
    synsets = {"test_object": {"coco_cat_id": 1, "synset": "test.n.01"},
               "other": {"coco_cat_id": 2, "synset": "other.n.01"}}
    review = draft_category_review(coco["categories"], lvis["categories"], synsets)
    for row in review["categories"]:
        row.update(reviewed=True, review_notes="Fixture category decision")
        if row["novelty"] is None:
            row["novelty"] = "unseen"
    return review


def test_draft_does_not_infer_unseen_from_unmatched_names():
    review = reviewed_fixture()
    coco = coco_fixture()
    draft = draft_category_review(coco["categories"], lvis_fixture()["categories"], {
        "test_object": {"coco_cat_id": 1, "synset": "test.n.01"},
        "other": {"coco_cat_id": 2, "synset": "other.n.01"},
    })
    assert draft["categories"][0]["novelty"] == "seen"
    assert draft["categories"][0]["coco_category_ids"] == [1]
    assert draft["categories"][1]["novelty"] is None
    with pytest.raises(ValueError, match="reviewed"):
        validate_category_review(draft, coco["categories"], lvis_fixture()["categories"])
    assert validate_category_review(review, coco["categories"], lvis_fixture()["categories"]) == {
        1: "seen", 2: "unseen"}


@pytest.mark.parametrize("mutation", [
    lambda r: r["categories"].pop(),
    lambda r: r["categories"].append(deepcopy(r["categories"][0])),
    lambda r: r["categories"][0].update(synset="stale.n.01"),
    lambda r: r["categories"][0].update(coco_category_ids=[999]),
    lambda r: r["categories"][0].update(novelty="unseen"),
    lambda r: r["categories"][1].update(novelty="seen"),
    lambda r: r["categories"][1].update(review_notes=" "),
    lambda r: r["categories"][1].update(review_notes=None),
    lambda r: r["categories"][1].update(reviewed="true"),
])
def test_incomplete_or_stale_decisions_rejected(mutation):
    review = reviewed_fixture()
    mutation(review)
    with pytest.raises(ValueError):
        validate_category_review(review, coco_fixture()["categories"], lvis_fixture()["categories"])


def test_lvis_preserves_geometry_filters_namespace_and_first_synonym():
    lvis = lvis_fixture()
    records = list(build_lvis_records(lvis, {1: "seen", 2: "unseen"}))
    reference = list(build_coco_records(coco_fixture()))
    assert [r.obj_id for r in records] == ["lvis:" + r.obj_id[5:] for r in reference]
    for actual, expected in zip(records, reference, strict=True):
        assert actual.source == "lvis" and actual.image_id == expected.image_id
        assert actual.category_name == ("first synonym" if actual.category_id == 1 else "novel")
        assert actual.novelty == ("seen" if actual.category_id == 1 else "unseen")
        assert actual.gt_box == expected.gt_box
        assert (decode_rle(actual.gt_mask) == decode_rle(expected.gt_mask)).all()
        assert actual.thinness == expected.thinness
        assert actual.n_neighbors == expected.n_neighbors
    assert list(build_lvis_records(lvis, {1: "seen", 2: "unseen"}, [])) == []
    with pytest.raises(ValueError, match="every LVIS"):
        list(build_lvis_records(lvis, {1: "seen"}))


def test_missing_first_synonym_and_incomplete_synsets_rejected():
    lvis = lvis_fixture()
    lvis["categories"][0]["synonyms"] = []
    with pytest.raises(ValueError, match="synonym"):
        list(build_lvis_records(lvis, {1: "seen", 2: "unseen"}))
    with pytest.raises(ValueError, match="exactly"):
        draft_category_review(coco_fixture()["categories"], lvis["categories"], {})
