"""Registry of `method` names allowed in ResultRow (docs/interfaces.md section 5). Add new methods by PR."""

import re

METHODS = {
    "b0_none": "no correction (step 0 only)",
    "b1_random_k": "k random positive clicks inside the box",
    "b2_multimask": "SAM multimask output, keep highest predicted IoU",
    "b3_box_refit": "refit the box to the current mask and rerun",
    "b3_unc_click": "click at the centre of the largest low-|logit| region",
    "oracle_clicks": "ideal clicks from GT (upper bound)",
}
_CORRECTOR = re.compile(r"corrector_v\d+")


def check_method(name: str) -> str:
    if name not in METHODS and not _CORRECTOR.fullmatch(name):
        raise ValueError(f"unknown method {name!r}; add it to promptcorrect/eval/methods.py")
    return name
