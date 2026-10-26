"""Owner: P4 (Aman). Rule-based baselines using only the Segmenter interface (see baselines/rules.py).

Method names are P3's registry: b0_none, b1_random_k, b2_multimask, b3_box_refit, b3_unc_click.
"""

from promptcorrect.baselines.rules import METHODS, RULES, Step, rollout

__all__ = ["METHODS", "RULES", "Step", "rollout"]
