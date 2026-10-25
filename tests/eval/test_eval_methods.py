import pytest

from promptcorrect.eval.methods import check_method


def test_registry():
    assert check_method("oracle_clicks") == "oracle_clicks"
    assert check_method("corrector_v12") == "corrector_v12"
    for bad in ("oracle", "corrector_vx", "corrector_v1_extra"):
        with pytest.raises(ValueError):
            check_method(bad)
