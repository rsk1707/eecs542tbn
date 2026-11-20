import pandas as pd
import pytest

from promptcorrect.eval.results import SCHEMA, read_results, update_runs_index, write_results
from promptcorrect.types import ResultRow


def _row(**kw):
    base = dict(run_id="demo_test_1", obj_id="coco:1", image_id=1, method="b0_none", step=0,
                iou=0.8, boundary_iou=0.5, prompt_origin="gt")
    return ResultRow(**{**base, **kw})


def test_round_trip_keeps_columns_dtypes_and_nulls(tmp_path):
    rows = [_row(), _row(method="oracle_clicks", step=1, iou=0.9, pert_type="shift", pert_level=0.2)]
    out = write_results(rows, tmp_path / "demo_test_1")
    df = read_results(out.parent)
    assert list(df.columns) == list(SCHEMA)
    assert df["step"].dtype == "int64" and df["iou"].dtype == "float64"
    assert df["pert_type"].isna().iloc[0] and df["pert_type"].iloc[1] == "shift"
    assert df["pert_level"].isna().iloc[0]  # None -> NaN for floats


def test_empty_run_still_has_schema(tmp_path):
    df = read_results(write_results([], tmp_path / "demo_empty"))
    assert df.empty and list(df.columns) == list(SCHEMA)


@pytest.mark.parametrize("rows, msg", [
    ([_row(method="my_method")], "unknown method"),
    ([_row(), _row()], "duplicate"),
    ([_row(), _row(run_id="other", obj_id="coco:2")], "one run per file"),
    ([_row(step=-1)], "step"),
])
def test_rejects_bad_rows(tmp_path, rows, msg):
    with pytest.raises(ValueError, match=msg):
        write_results(rows, tmp_path / "demo_test_1")


def test_run_id_must_match_dir_and_no_silent_overwrite(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        write_results([_row()], tmp_path / "wrong_dir")
    write_results([_row()], tmp_path / "demo_test_1")
    with pytest.raises(FileExistsError):
        write_results([_row()], tmp_path / "demo_test_1")
    write_results([_row(iou=0.1)], tmp_path / "demo_test_1", overwrite=True)
    assert read_results(tmp_path / "demo_test_1")["iou"].iloc[0] == 0.1


def test_runs_index_adds_and_replaces(tmp_path):
    idx = tmp_path / "runs_index.csv"
    update_runs_index("r1", "P3", "/runs/r1", headline={"miou": 0.8}, index_path=idx)
    update_runs_index("r2", "P4", "/runs/r2", index_path=idx)
    update_runs_index("r1", "P3", "/runs/r1", notes="rerun", headline={"miou": 0.85}, index_path=idx)
    df = pd.read_csv(idx)
    assert list(df["run_id"]) == ["r2", "r1"]
    assert df.set_index("run_id").loc["r1", "notes"] == "rerun"
    assert df.set_index("run_id").loc["r1", "miou"] == 0.85
