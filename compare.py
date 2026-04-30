"""
compare.py — Inter-annotator agreement (IAA) between human pilot annotations
and Claude's listening judgments.

Computes and reports:

Overall judgment (Pro / Con / Tie)
  - Raw accuracy
  - Cohen's kappa (3-class and 2-class, dropping Tie)
  - Gwet's AC1 (more robust than kappa when one class dominates)
  - Full confusion matrix + per-class precision / recall / F1
  - Bootstrap 95% CIs on the headline numbers

Per-dimension sub-scores (1-5 ordinal, 5 dims x {pro, con} = 10 cells)
  - Weighted Cohen's kappa (quadratic weights) with bootstrap CI
  - Spearman rho and Pearson r
  - MAE, RMSE, exact-match / within-1 / within-2
  - Bias (mean claude - human) per cell, to detect systematic skew
  - Pooled metrics across all 10 cells

Internal consistency
  - For each annotator, does their stated overall judgment match the one
    you'd get by taking the mean of their own sub-scores? Sanity check.

Outputs:
  reports/iaa/iaa_report.md              - full markdown report
  reports/iaa/iaa_per_dimension.csv      - per-cell metrics (for thesis tables)
  reports/iaa/iaa_disagreements.md       - every disagreement, side by side
  reports/iaa/iaa_heatmap.png            - per-dimension weighted-kappa heatmap
  reports/iaa/iaa_confusion_matrix.png   - 3×3 overall judgment confusion matrix

Usage:
    python compare.py                         # vs. claude_listening_trial.json (default)
    python compare.py --use-full              # vs. claude_listening.json
    python compare.py --bootstrap 2000        # bootstrap resamples
    python compare.py --output-dir reports/x  # custom dir
    python compare.py --no-plots              # skip matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    ANNOTATIONS_PATH,
    CLAUDE_LISTENING_PATH,
    CLAUDE_LISTENING_TRIAL_PATH,
    REPORTS_DIR,
)

LISTENING_DIMENSIONS = (
    "acknowledgment",
    "accuracy_of_representation",
    "responsiveness",
    "concession_and_common_ground",
    "respectful_engagement",
)
SHORT_DIMENSIONS = (
    "acknowledgment",
    "accuracy",
    "responsiveness",
    "concession",
    "respectful",
)
SIDES = ("pro", "con")
JUDGMENT_LABELS = ("Pro", "Con", "Tie")
# Per-dimension score ranges. concession_and_common_ground uses a narrower 1–3 scale
# because explicit concession is rare in online asynchronous debate; the pilot showed
# the additional 1–5 anchors were unreliable (human cluster at 1–2, Claude used full range).
DIMENSION_SCORE_RANGES: dict[str, tuple[int, int]] = {
    "acknowledgment": (1, 5),
    "accuracy_of_representation": (1, 5),
    "responsiveness": (1, 5),
    "concession_and_common_ground": (1, 3),
    "respectful_engagement": (1, 5),
}


# --------------------------- parsing helpers ---------------------------

def _extract_overall_judgment(record: dict) -> str | None:
    if not isinstance(record, dict):
        return None
    overall = record.get("overall_better_listener")
    if isinstance(overall, dict):
        j = overall.get("judgment")
        if j in JUDGMENT_LABELS:
            return j
    for key in ("overall_judgment", "judgment"):
        j = record.get(key)
        if j in JUDGMENT_LABELS:
            return j
    return None


def _extract_scores(record: dict) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    if not isinstance(record, dict):
        return out
    for side in SIDES:
        side_obj = record.get(side)
        if not isinstance(side_obj, dict):
            continue
        for dim in LISTENING_DIMENSIONS:
            dim_obj = side_obj.get(dim)
            if not isinstance(dim_obj, dict):
                continue
            score = dim_obj.get("score")
            min_val, max_val = DIMENSION_SCORE_RANGES[dim]
            if isinstance(score, int) and min_val <= score <= max_val:
                out[(side, dim)] = score
            elif isinstance(score, str) and score.isdigit() and min_val <= int(score) <= max_val:
                out[(side, dim)] = int(score)
    return out


def _extract_justifications(record: dict) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    if not isinstance(record, dict):
        return out
    for side in SIDES:
        side_obj = record.get(side)
        if not isinstance(side_obj, dict):
            continue
        for dim in LISTENING_DIMENSIONS:
            dim_obj = side_obj.get(dim)
            if isinstance(dim_obj, dict):
                out[(side, dim)] = str(dim_obj.get("justification", "")).strip()
    return out


def _extract_overall_justification(record: dict) -> str:
    if not isinstance(record, dict):
        return ""
    overall = record.get("overall_better_listener")
    if isinstance(overall, dict):
        return str(overall.get("justification", "")).strip()
    return ""


def _load_human(path: Path) -> dict[int, dict]:
    """Return {debate_id: listening_record}. If multiple per debate, take the last."""
    with open(path) as f:
        raw = json.load(f)
    out: dict[int, dict] = {}
    for did_str, tasks in raw.items():
        if not isinstance(tasks, dict):
            continue
        listening = tasks.get("listening")
        if isinstance(listening, list):
            listening = listening[-1] if listening else None
        if isinstance(listening, dict):
            out[int(did_str)] = listening
    return out


def _load_claude(path: Path) -> dict[int, dict]:
    """Return {debate_id: evaluation_record}. If multiple, take the last."""
    with open(path) as f:
        raw = json.load(f)
    out: dict[int, dict] = {}
    for r in raw:
        if not isinstance(r, dict) or "debate_id" not in r:
            continue
        did = int(r["debate_id"])
        evaluation = r.get("evaluation") if isinstance(r.get("evaluation"), dict) else r
        out[did] = evaluation
    return out


# --------------------------- metric implementations ---------------------------

def cohen_kappa(y1: Sequence, y2: Sequence, labels: Sequence | None = None) -> float:
    """Unweighted Cohen's kappa for nominal data."""
    y1 = list(y1)
    y2 = list(y2)
    n = len(y1)
    if n == 0:
        return float("nan")
    if labels is None:
        labels = sorted(set(y1) | set(y2))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        if a not in label_to_idx or b not in label_to_idx:
            continue
        cm[label_to_idx[a], label_to_idx[b]] += 1
    total = cm.sum()
    if total == 0:
        return float("nan")
    po = np.trace(cm) / total
    row = cm.sum(axis=1) / total
    col = cm.sum(axis=0) / total
    pe = float(np.sum(row * col))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return (po - pe) / (1 - pe)


def weighted_kappa(y1: Sequence[int], y2: Sequence[int],
                   min_val: int, max_val: int,
                   weights: str = "quadratic") -> float:
    """Weighted Cohen's kappa (linear or quadratic) for ordinal data."""
    y1 = np.asarray(list(y1), dtype=float)
    y2 = np.asarray(list(y2), dtype=float)
    n = len(y1)
    if n == 0:
        return float("nan")
    k = max_val - min_val + 1
    cm = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        cm[int(a) - min_val, int(b) - min_val] += 1
    row_marg = cm.sum(axis=1)
    col_marg = cm.sum(axis=0)
    expected = np.outer(row_marg, col_marg) / n
    w = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if weights == "quadratic":
                w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)
            else:  # linear
                w[i, j] = abs(i - j) / (k - 1)
    num = float(np.sum(w * cm))
    den = float(np.sum(w * expected))
    if den == 0:
        return 1.0 if num == 0 else float("nan")
    return 1 - num / den


def krippendorff_alpha_ordinal(y1: Sequence[int], y2: Sequence[int],
                                min_val: int, max_val: int) -> float:
    """Krippendorff's alpha with ordinal (interval-difference) weights.

    Complementary to weighted kappa; more stable than kappa when n is small
    or the score distribution is restricted. Uses the standard formulation
    from Krippendorff (2011): alpha = 1 - (D_o / D_e) where D_o is observed
    disagreement and D_e is expected disagreement, both computed with
    squared-difference weights for ordinal data.
    """
    y1 = list(y1)
    y2 = list(y2)
    n = len(y1)
    if n == 0:
        return float("nan")
    # Build the coincidence-style sums directly for the 2-coder case.
    # Observed disagreement: mean squared diff over pairs.
    d_o = float(np.mean([(a - b) ** 2 for a, b in zip(y1, y2)]))
    # Expected disagreement: mean squared diff over all possible cross-pairs
    # of the combined marginal distribution.
    combined = y1 + y2
    N = len(combined)
    if N < 2:
        return float("nan")
    total_sq = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            total_sq += (combined[i] - combined[j]) ** 2
    d_e = total_sq / (N * (N - 1))
    if d_e == 0:
        return 1.0 if d_o == 0 else float("nan")
    return 1 - (d_o / d_e)


def gwet_ac1(y1: Sequence, y2: Sequence, labels: Sequence | None = None) -> float:
    """Gwet's AC1 — more robust than kappa when one class dominates."""
    y1 = list(y1)
    y2 = list(y2)
    n = len(y1)
    if n == 0:
        return float("nan")
    if labels is None:
        labels = sorted(set(y1) | set(y2))
    k = len(labels)
    if k < 2:
        return float("nan")
    label_to_idx = {l: i for i, l in enumerate(labels)}
    agree = sum(1 for a, b in zip(y1, y2) if a == b) / n
    p = np.zeros(k)
    for a, b in zip(y1, y2):
        if a in label_to_idx:
            p[label_to_idx[a]] += 1
        if b in label_to_idx:
            p[label_to_idx[b]] += 1
    p /= (2 * n)
    pe = float(np.sum(p * (1 - p)) / (k - 1))
    if pe >= 1.0:
        return 1.0 if agree == 1.0 else float("nan")
    return (agree - pe) / (1 - pe)


def confusion_matrix(y1: Sequence, y2: Sequence, labels: Sequence) -> np.ndarray:
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(y1, y2):
        if a in label_to_idx and b in label_to_idx:
            cm[label_to_idx[a], label_to_idx[b]] += 1
    return cm


def per_class_prf(cm: np.ndarray, labels: Sequence) -> dict:
    """Given confusion matrix with rows=reference (human), cols=predicted (Claude)."""
    out = {}
    for i, l in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else float("nan")
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        f = 2 * p * r / (p + r) if (p + r) and not (math.isnan(p) or math.isnan(r)) else float("nan")
        out[l] = {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(cm[i, :].sum())}
    return out


def bootstrap_ci(
    value_pairs: list[tuple],
    stat_fn: Callable[[list, list], float],
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap percentile CI over a list of (a, b) pairs."""
    if not value_pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(value_pairs)
    samples = []
    for _ in range(n_boot):
        draw = [value_pairs[rng.randrange(n)] for _ in range(n)]
        a = [p[0] for p in draw]
        b = [p[1] for p in draw]
        try:
            v = stat_fn(a, b)
            if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                samples.append(v)
        except Exception:
            pass
    if not samples:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(samples, (1 - ci) / 2 * 100))
    hi = float(np.percentile(samples, (1 + ci) / 2 * 100))
    return (lo, hi)


def _derived_overall_from_scores(scores: dict[tuple[str, str], int]) -> str | None:
    """Take mean across 5 dims per side, higher mean = better listener.

    Mixing scales (concession 1–3, others 1–5) is acceptable here because both Pro and
    Con use the same per-dimension scale, so the relative comparison within a debate is
    scale-invariant.
    """
    pro_vals = [v for (side, _), v in scores.items() if side == "pro"]
    con_vals = [v for (side, _), v in scores.items() if side == "con"]
    if len(pro_vals) != len(LISTENING_DIMENSIONS) or len(con_vals) != len(LISTENING_DIMENSIONS):
        return None
    p, c = float(np.mean(pro_vals)), float(np.mean(con_vals))
    if abs(p - c) < 1e-9:
        return "Tie"
    return "Pro" if p > c else "Con"


# --------------------------- analysis ---------------------------

def analyze(human: dict[int, dict], claude: dict[int, dict], n_boot: int) -> dict:
    overlap = sorted(set(human) & set(claude))
    result: dict[str, Any] = {
        "n_human_debates": len(human),
        "n_claude_debates": len(claude),
        "n_overlap": len(overlap),
        "overlap_ids": overlap,
    }

    h_overall: list[str] = []
    c_overall: list[str] = []
    dim_pairs: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    h_derived: list[str] = []
    c_derived: list[str] = []

    for did in overlap:
        h_rec = human[did]
        c_rec = claude[did]
        h_j = _extract_overall_judgment(h_rec)
        c_j = _extract_overall_judgment(c_rec)
        if h_j and c_j:
            h_overall.append(h_j)
            c_overall.append(c_j)
        h_scores = _extract_scores(h_rec)
        c_scores = _extract_scores(c_rec)
        for side in SIDES:
            for dim in LISTENING_DIMENSIONS:
                key = (side, dim)
                if key in h_scores and key in c_scores:
                    dim_pairs[key].append((h_scores[key], c_scores[key]))
        hd = _derived_overall_from_scores(h_scores)
        cd = _derived_overall_from_scores(c_scores)
        if hd and cd:
            h_derived.append(hd)
            c_derived.append(cd)

    result["overall"] = _analyze_overall(h_overall, c_overall, n_boot)
    result["overall_derived"] = _analyze_overall(h_derived, c_derived, n_boot)
    result["dimensions"] = _analyze_dimensions(dim_pairs, n_boot)
    result["internal_consistency"] = _internal_consistency(overlap, human, claude)
    return result


def _analyze_overall(h: list[str], c: list[str], n_boot: int) -> dict:
    if not h:
        return {"n": 0}
    n = len(h)
    agree = sum(1 for a, b in zip(h, c) if a == b)
    accuracy = agree / n

    cm3 = confusion_matrix(h, c, JUDGMENT_LABELS)
    k3 = cohen_kappa(h, c, JUDGMENT_LABELS)
    ac1_3 = gwet_ac1(h, c, JUDGMENT_LABELS)
    prf3 = per_class_prf(cm3, JUDGMENT_LABELS)

    pairs_2 = [(a, b) for a, b in zip(h, c) if a != "Tie" and b != "Tie"]
    if pairs_2:
        h2 = [p[0] for p in pairs_2]
        c2 = [p[1] for p in pairs_2]
        acc2 = sum(1 for a, b in pairs_2 if a == b) / len(pairs_2)
        cm2 = confusion_matrix(h2, c2, ("Pro", "Con"))
        k2 = cohen_kappa(h2, c2, ("Pro", "Con"))
        ac1_2 = gwet_ac1(h2, c2, ("Pro", "Con"))
    else:
        acc2 = float("nan")
        cm2 = None
        k2 = float("nan")
        ac1_2 = float("nan")

    pairs = list(zip(h, c))
    acc_ci = bootstrap_ci(
        pairs,
        lambda a, b: sum(1 for x, y in zip(a, b) if x == y) / len(a),
        n_boot,
    )
    k3_ci = bootstrap_ci(pairs, lambda a, b: cohen_kappa(a, b, JUDGMENT_LABELS), n_boot)
    ac1_ci = bootstrap_ci(pairs, lambda a, b: gwet_ac1(a, b, JUDGMENT_LABELS), n_boot)

    return {
        "n": n,
        "accuracy": accuracy,
        "accuracy_ci": acc_ci,
        "cohen_kappa_3class": k3,
        "cohen_kappa_3class_ci": k3_ci,
        "gwet_ac1_3class": ac1_3,
        "gwet_ac1_3class_ci": ac1_ci,
        "confusion_matrix_3class": cm3.tolist(),
        "per_class_prf_3class": prf3,
        "n_2class": len(pairs_2),
        "accuracy_2class": acc2,
        "cohen_kappa_2class": k2,
        "gwet_ac1_2class": ac1_2,
        "confusion_matrix_2class": cm2.tolist() if cm2 is not None else None,
    }


def _analyze_dimensions(dim_pairs: dict, n_boot: int) -> dict:
    rows = []
    for side in SIDES:
        for dim in LISTENING_DIMENSIONS:
            key = (side, dim)
            pairs = dim_pairs.get(key, [])
            min_val, max_val = DIMENSION_SCORE_RANGES[dim]
            if not pairs:
                rows.append({"side": side, "dimension": dim, "n": 0, "scale_max": max_val})
                continue
            h = np.array([p[0] for p in pairs])
            c = np.array([p[1] for p in pairs])
            diffs = c - h
            abs_diffs = np.abs(diffs)

            exact = float(np.mean(abs_diffs == 0))
            within1 = float(np.mean(abs_diffs <= 1))
            within2 = float(np.mean(abs_diffs <= 2))
            mae = float(np.mean(abs_diffs))
            rmse = float(np.sqrt(np.mean(diffs ** 2)))
            bias = float(np.mean(diffs))

            wk = weighted_kappa(h.tolist(), c.tolist(), min_val, max_val)
            try:
                spearman_r = float(stats.spearmanr(h, c).correlation)
            except Exception:
                spearman_r = float("nan")
            try:
                pearson_r = float(stats.pearsonr(h, c)[0])
            except Exception:
                pearson_r = float("nan")

            # Gwet's AC1 — robust under restricted range / skewed marginals.
            # Treat each ordinal level as a nominal label for AC1; this is the standard
            # fallback when κ is deflated by a narrow score distribution.
            ac1 = gwet_ac1(h.tolist(), c.tolist(), labels=list(range(min_val, max_val + 1)))

            # Krippendorff's α (ordinal) — complementary to weighted κ, more stable on small n.
            alpha = krippendorff_alpha_ordinal(h.tolist(), c.tolist(), min_val, max_val)

            ac1_ci = bootstrap_ci(
                pairs,
                lambda a, b, _min=min_val, _max=max_val: gwet_ac1(a, b, labels=list(range(_min, _max + 1))),
                n_boot,
            )

            # Use default-argument capture to avoid late-binding closure gotcha.
            wk_ci = bootstrap_ci(
                pairs,
                lambda a, b, _min=min_val, _max=max_val: weighted_kappa(a, b, _min, _max),
                n_boot,
            )
            mae_ci = bootstrap_ci(
                pairs,
                lambda a, b: float(np.mean(np.abs(np.array(a) - np.array(b)))),
                n_boot,
            )

            rows.append({
                "side": side,
                "dimension": dim,
                "n": len(pairs),
                "scale_max": max_val,
                "mae": mae,
                "mae_ci_lo": mae_ci[0],
                "mae_ci_hi": mae_ci[1],
                "rmse": rmse,
                "bias_claude_minus_human": bias,
                "exact_pct": exact,
                "within_1_pct": within1,
                "within_2_pct": within2,
                "weighted_kappa_quadratic": wk,
                "wk_ci_lo": wk_ci[0],
                "wk_ci_hi": wk_ci[1],
                "spearman_r": spearman_r,
                "pearson_r": pearson_r,
                "gwet_ac1": ac1,
                "ac1_ci_lo": ac1_ci[0],
                "ac1_ci_hi": ac1_ci[1],
                "krippendorff_alpha_ordinal": alpha,
            })

    # Pooled metrics over 1–5 dimensions only. concession_and_common_ground is on a
    # 1–3 scale; including it in a pooled weighted-κ would distort the weight matrix.
    pooled_1_5_pairs = [
        p
        for (side, dim), pairs in dim_pairs.items()
        for p in pairs
        if DIMENSION_SCORE_RANGES[dim] == (1, 5)
    ]
    pooled: dict = {}
    if pooled_1_5_pairs:
        h = np.array([p[0] for p in pooled_1_5_pairs])
        c = np.array([p[1] for p in pooled_1_5_pairs])
        pooled = {
            "n": len(pooled_1_5_pairs),
            "weighted_kappa_quadratic": weighted_kappa(h.tolist(), c.tolist(), 1, 5),
            "mae": float(np.mean(np.abs(h - c))),
            "rmse": float(np.sqrt(np.mean((h - c) ** 2))),
            "exact_pct": float(np.mean(h == c)),
            "within_1_pct": float(np.mean(np.abs(h - c) <= 1)),
            "bias": float(np.mean(c - h)),
        }

    # Concession (1–3 scale) pooled separately.
    concession_pairs = [
        p
        for (side, dim), pairs in dim_pairs.items()
        for p in pairs
        if dim == "concession_and_common_ground"
    ]
    concession_pooled: dict = {}
    if concession_pairs:
        h_c = np.array([p[0] for p in concession_pairs])
        c_c = np.array([p[1] for p in concession_pairs])
        concession_pooled = {
            "n": len(concession_pairs),
            "weighted_kappa_quadratic": weighted_kappa(h_c.tolist(), c_c.tolist(), 1, 3),
            "mae": float(np.mean(np.abs(h_c - c_c))),
            "rmse": float(np.sqrt(np.mean((h_c - c_c) ** 2))),
            "exact_pct": float(np.mean(h_c == c_c)),
            "within_1_pct": float(np.mean(np.abs(h_c - c_c) <= 1)),
            "bias": float(np.mean(c_c - h_c)),
        }

    return {"per_cell": rows, "pooled": pooled, "concession_pooled": concession_pooled}


def _internal_consistency(overlap: list[int], human: dict, claude: dict) -> dict:
    """Does each annotator's stated overall agree with the mean-of-sub-scores derived one?"""
    out = {}
    for name, source in [("human", human), ("claude", claude)]:
        stated = []
        derived = []
        for did in overlap:
            rec = source[did]
            s = _extract_overall_judgment(rec)
            d = _derived_overall_from_scores(_extract_scores(rec))
            if s and d:
                stated.append(s)
                derived.append(d)
        if stated:
            acc = sum(1 for a, b in zip(stated, derived) if a == b) / len(stated)
            k = cohen_kappa(stated, derived, JUDGMENT_LABELS)
            out[name] = {"n": len(stated), "accuracy": acc, "cohen_kappa": k}
        else:
            out[name] = {"n": 0}
    return out


# --------------------------- reporting ---------------------------

def _fmt(x, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _fmt_ci(ci, digits: int = 3) -> str:
    if ci is None:
        return "—"
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return "—"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def write_report(result: dict, output_dir: Path, make_plots: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Inter-annotator agreement report")
    lines.append("")
    lines.append(f"- Human annotations: **{result['n_human_debates']}** debates")
    lines.append(f"- Claude annotations: **{result['n_claude_debates']}** debates")
    lines.append(f"- Overlap (used for IAA): **{result['n_overlap']}** debates")
    lines.append("")

    # Overall judgment (stated)
    o = result["overall"]
    lines.append("## Overall judgment (Pro / Con / Tie) — stated")
    lines.append("")
    if o["n"] == 0:
        lines.append("_No overlapping overall judgments._")
    else:
        lines.append(f"- n = {o['n']}")
        lines.append(f"- Raw accuracy: **{_fmt(o['accuracy'])}**  CI {_fmt_ci(o['accuracy_ci'])}")
        lines.append(f"- Cohen's κ (3-class): **{_fmt(o['cohen_kappa_3class'])}**  CI {_fmt_ci(o['cohen_kappa_3class_ci'])}")
        lines.append(f"- Gwet's AC1 (3-class): **{_fmt(o['gwet_ac1_3class'])}**  CI {_fmt_ci(o['gwet_ac1_3class_ci'])}")
        lines.append("")
        lines.append("**Confusion matrix (rows = human, cols = Claude):**")
        lines.append("")
        cm = o["confusion_matrix_3class"]
        lines.append("| human \\ Claude | Pro | Con | Tie | total |")
        lines.append("|---|---|---|---|---|")
        for i, lbl in enumerate(JUDGMENT_LABELS):
            total = sum(cm[i])
            lines.append(f"| **{lbl}** | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} | {total} |")
        col_tot = [sum(cm[i][j] for i in range(3)) for j in range(3)]
        lines.append(f"| **total** | {col_tot[0]} | {col_tot[1]} | {col_tot[2]} | {o['n']} |")
        lines.append("")
        lines.append("**Per-class (human as reference):**")
        lines.append("")
        lines.append("| class | precision | recall | F1 | support |")
        lines.append("|---|---|---|---|---|")
        for lbl in JUDGMENT_LABELS:
            pr = o["per_class_prf_3class"][lbl]
            lines.append(f"| {lbl} | {_fmt(pr['precision'])} | {_fmt(pr['recall'])} | {_fmt(pr['f1'])} | {pr['support']} |")
        lines.append("")
        if o.get("n_2class", 0):
            lines.append("### 2-class variant (drop Tie debates)")
            lines.append("")
            lines.append(f"- n = {o['n_2class']}")
            lines.append(f"- Accuracy: {_fmt(o['accuracy_2class'])}")
            lines.append(f"- Cohen's κ: {_fmt(o['cohen_kappa_2class'])}")
            lines.append(f"- Gwet's AC1: {_fmt(o['gwet_ac1_2class'])}")
            lines.append("")

    # Derived overall
    od = result["overall_derived"]
    if od.get("n"):
        lines.append("## Overall judgment — derived from mean sub-scores")
        lines.append("")
        lines.append("_Each annotator's overall judgment redefined as 'whichever side has the higher mean across the 5 sub-dims'. A sanity check that sub-scores line up with the stated overall._")
        lines.append("")
        lines.append(f"- n = {od['n']}")
        lines.append(f"- Accuracy: **{_fmt(od['accuracy'])}**  CI {_fmt_ci(od['accuracy_ci'])}")
        lines.append(f"- Cohen's κ (3-class): **{_fmt(od['cohen_kappa_3class'])}**  CI {_fmt_ci(od['cohen_kappa_3class_ci'])}")
        lines.append("")

    # Internal consistency
    ic = result["internal_consistency"]
    lines.append("## Internal consistency (stated vs. derived)")
    lines.append("")
    lines.append("_How often each annotator's stated overall judgment agrees with the one computed from the mean of their own sub-scores._")
    lines.append("")
    lines.append("| annotator | n | accuracy | Cohen's κ |")
    lines.append("|---|---|---|---|")
    for name in ("human", "claude"):
        rec = ic.get(name, {})
        if rec.get("n"):
            lines.append(f"| {name} | {rec['n']} | {_fmt(rec['accuracy'])} | {_fmt(rec['cohen_kappa'])} |")
        else:
            lines.append(f"| {name} | 0 | — | — |")
    lines.append("")

    # Per-dimension
    dims = result["dimensions"]
    lines.append("## Per-dimension agreement")
    lines.append("")
    pooled = dims.get("pooled", {})
    concession_pooled = dims.get("concession_pooled", {})
    if pooled:
        lines.append(f"**Pooled across the four 1–5 dimensions ({pooled['n']} cell observations; concession_and_common_ground excluded — see note below):**")
        lines.append("")
        lines.append(f"- weighted κ (quadratic): **{_fmt(pooled['weighted_kappa_quadratic'])}**")
        lines.append(f"- MAE: {_fmt(pooled['mae'])}, RMSE: {_fmt(pooled['rmse'])}")
        lines.append(f"- exact match: {_fmt(pooled['exact_pct'])}, within-1: {_fmt(pooled['within_1_pct'])}")
        lines.append(f"- pooled bias (Claude − human): {_fmt(pooled['bias'])}")
        lines.append("")
    if concession_pooled:
        lines.append(f"**concession_and_common_ground (1–3 scale, {concession_pooled['n']} cell observations, reported separately):**")
        lines.append("")
        lines.append(f"- weighted κ (quadratic): **{_fmt(concession_pooled['weighted_kappa_quadratic'])}**")
        lines.append(f"- MAE: {_fmt(concession_pooled['mae'])}, RMSE: {_fmt(concession_pooled['rmse'])}")
        lines.append(f"- exact match: {_fmt(concession_pooled['exact_pct'])}, within-1: {_fmt(concession_pooled['within_1_pct'])}")
        lines.append(f"- bias (Claude − human): {_fmt(concession_pooled['bias'])}")
        lines.append("")
    lines.append("| side | dimension | n | wκ | AC1 | α (ord) | Spearman | MAE | bias | exact | ≤1 | ≤2 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in dims["per_cell"]:
        if row["n"] == 0:
            lines.append(f"| {row['side']} | {row['dimension']} | 0 | — | — | — | — | — | — | — | — | — | — |")
            continue
        # On a 1–3 scale, within-2 is trivially 100%; replace with a dash for clarity.
        within2_col = "—" if row.get("scale_max", 5) <= 3 else _fmt(row["within_2_pct"])
        lines.append(
            f"| {row['side']} | {row['dimension']} | {row['n']} | "
            f"{_fmt(row['weighted_kappa_quadratic'])} | "
            f"{_fmt(row['gwet_ac1'])} | "
            f"{_fmt(row['krippendorff_alpha_ordinal'])} | "
            f"{_fmt(row['spearman_r'])} | "
            f"{_fmt(row['mae'])} | "
            f"{_fmt(row['bias_claude_minus_human'])} | "
            f"{_fmt(row['exact_pct'])} | "
            f"{_fmt(row['within_1_pct'])} | "
            f"{within2_col} |"
        )
    lines.append("")

    lines.append("### How to read this")
    lines.append("")
    lines.append("- **weighted κ (quadratic)** is the primary metric for ordinal rubric scores. Rough Landis & Koch (1977) guide: 0.0–0.2 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.0 almost perfect.")
    lines.append("- **Gwet's AC1** is reported alongside Cohen's κ because κ is deflated when one class dominates (the 'high agreement, low κ' paradox). AC1 is more stable under skewed marginals.")
    lines.append(
        "- **Krippendorff's α (ordinal)** is reported alongside weighted κ as a second ordinal-agreement metric. When a dimension has a compressed score range (e.g., only 3s and 4s are used), weighted κ can be deflated even though annotators largely agree; α and AC1 are less sensitive to this failure mode. Report all three in the thesis so the reader can see whether a low κ is genuine disagreement or restricted-range deflation.")
    lines.append("- **bias (Claude − human)** — positive means Claude rates higher on average than you do on that cell. Large per-dimension biases are a signal that the rubric wording or the prompt needs tightening for that dimension.")
    lines.append("- **exact / ≤1 / ≤2** is the forgiving-agreement ladder. On a 1–5 scale, within-1 ≈ 0.80+ is usually considered strong for ordinal rubrics.")
    lines.append("- **Pooled metrics exclude concession_and_common_ground** because that dimension uses a 1–3 scale while the other four use 1–5. Mixing scales in a pooled weighted κ would distort the weight matrix. Concession metrics are reported separately in the pooled-summary section above.")
    lines.append("- All CIs are non-parametric bootstrap percentile intervals resampled over debates.")
    lines.append("")

    (output_dir / "iaa_report.md").write_text("\n".join(lines) + "\n")

    # CSV
    df = pd.DataFrame(dims["per_cell"])
    df.to_csv(output_dir / "iaa_per_dimension.csv", index=False)

    # Heatmaps
    if make_plots:
        _write_heatmap(dims["per_cell"], output_dir)
        if o.get("n", 0) > 0:
            cm3 = np.array(o["confusion_matrix_3class"])
            _write_confusion_heatmap(cm3, output_dir)


def write_disagreements(human: dict, claude: dict, output_dir: Path) -> None:
    overlap = sorted(set(human) & set(claude))
    lines: list[str] = ["# Disagreements — human vs Claude", ""]
    body: list[str] = []
    count = 0
    for did in overlap:
        h = human[did]
        c = claude[did]
        h_j = _extract_overall_judgment(h)
        c_j = _extract_overall_judgment(c)
        h_scores = _extract_scores(h)
        c_scores = _extract_scores(c)
        h_just = _extract_justifications(h)
        c_just = _extract_justifications(c)
        h_overall_just = _extract_overall_justification(h)
        c_overall_just = _extract_overall_justification(c)

        overall_disagree = bool(h_j and c_j and h_j != c_j)
        score_disagreements = []
        for side in SIDES:
            for dim in LISTENING_DIMENSIONS:
                key = (side, dim)
                if key in h_scores and key in c_scores:
                    diff = c_scores[key] - h_scores[key]
                    if abs(diff) >= 2:
                        score_disagreements.append((side, dim, h_scores[key], c_scores[key], diff))

        if not overall_disagree and not score_disagreements:
            continue
        count += 1
        body.append(f"## Debate {did}")
        body.append("")
        flag = "  ⚠️ DISAGREE" if overall_disagree else ""
        body.append(f"**Overall:** human = `{h_j}`, Claude = `{c_j}`{flag}")
        body.append("")
        if h_overall_just:
            body.append(f"> *human:* {h_overall_just}")
        if c_overall_just:
            body.append(f"> *Claude:* {c_overall_just}")
        body.append("")
        if score_disagreements:
            body.append("**Sub-score disagreements (|diff| ≥ 2):**")
            body.append("")
            body.append("| side | dimension | human | Claude | diff |")
            body.append("|---|---|---|---|---|")
            for side, dim, hs, cs, d in score_disagreements:
                body.append(f"| {side} | {dim} | {hs} | {cs} | {d:+d} |")
            body.append("")
            for side, dim, *_ in score_disagreements:
                cj = c_just.get((side, dim), "")
                if cj:
                    body.append(f"- *Claude on {side}.{dim}:* {cj}")
                hj = h_just.get((side, dim), "")
                if hj:
                    body.append(f"- *human on {side}.{dim}:* {hj}")
            body.append("")

    lines.append(f"**Total debates with disagreements:** {count} / {len(overlap)}")
    lines.append("")
    lines.extend(body)
    (output_dir / "iaa_disagreements.md").write_text("\n".join(lines) + "\n")


def _write_heatmap(per_cell: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping heatmap figure.")
        return

    mat = np.full((len(LISTENING_DIMENSIONS), len(SIDES)), np.nan)
    for row in per_cell:
        if row["n"] == 0:
            continue
        i = LISTENING_DIMENSIONS.index(row["dimension"])
        j = SIDES.index(row["side"])
        mat[i, j] = row["weighted_kappa_quadratic"]

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    im = ax.imshow(mat, vmin=-0.2, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(SIDES)))
    ax.set_xticklabels([s.capitalize() for s in SIDES])
    ax.set_yticks(range(len(LISTENING_DIMENSIONS)))
    ax.set_yticklabels([d.replace("_", " ") for d in SHORT_DIMENSIONS])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not math.isnan(v):
                color = "black" if 0.2 < v < 0.8 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=11)
    ax.set_title("Weighted κ (quadratic) by dimension × side")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "iaa_heatmap.png", dpi=150)
    plt.close(fig)


def _write_confusion_heatmap(cm: np.ndarray, output_dir: Path) -> None:
    """Generate a 3×3 confusion matrix heatmap for overall better-listener judgment."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping confusion matrix heatmap.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(JUDGMENT_LABELS)
    ax.set_yticks(range(3))
    ax.set_yticklabels(JUDGMENT_LABELS)

    ax.set_xlabel("Claude's Judgment", fontsize=12)
    ax.set_ylabel("Human Judgment", fontsize=12)
    ax.set_title("Overall Better-Listener Judgment Confusion Matrix", fontsize=13, pad=15)

    # Annotate cells with counts
    for i in range(3):
        for j in range(3):
            count = cm[i, j]
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(count), ha="center", va="center",
                   color=text_color, fontsize=14, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "iaa_confusion_matrix.png", dpi=150)
    plt.close(fig)


# --------------------------- main ---------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="IAA: human vs Claude listening annotations")
    parser.add_argument("--use-full", action="store_true",
                        help="Compare against claude_listening.json instead of claude_listening_trial.json")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {REPORTS_DIR}/iaa)")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib figures")
    args = parser.parse_args()

    if not ANNOTATIONS_PATH.exists():
        print(f"No annotations file: {ANNOTATIONS_PATH}")
        return 1
    claude_path = CLAUDE_LISTENING_PATH if args.use_full else CLAUDE_LISTENING_TRIAL_PATH
    if not claude_path.exists():
        print(f"No Claude output: {claude_path}")
        return 1

    human = _load_human(ANNOTATIONS_PATH)
    claude = _load_claude(claude_path)

    output_dir = Path(args.output_dir) if args.output_dir else (REPORTS_DIR / "iaa")

    print(f"Human: {len(human)} debates | Claude: {len(claude)} debates | "
          f"Overlap: {len(set(human) & set(claude))}")
    if not (set(human) & set(claude)):
        print("No overlap. Nothing to compare.")
        return 1

    result = analyze(human, claude, n_boot=args.bootstrap)
    write_report(result, output_dir, make_plots=not args.no_plots)
    write_disagreements(human, claude, output_dir)

    o = result["overall"]
    print()
    print("Headline numbers:")
    print(f"  overall accuracy:  {_fmt(o.get('accuracy'))} {_fmt_ci(o.get('accuracy_ci'))}")
    print(f"  Cohen κ (3-class): {_fmt(o.get('cohen_kappa_3class'))} {_fmt_ci(o.get('cohen_kappa_3class_ci'))}")
    print(f"  Gwet AC1:          {_fmt(o.get('gwet_ac1_3class'))}")
    pooled = result["dimensions"].get("pooled", {})
    if pooled:
        print(f"  pooled weighted κ (1–5 dims): {_fmt(pooled.get('weighted_kappa_quadratic'))}")
        print(f"  pooled within-1   (1–5 dims): {_fmt(pooled.get('within_1_pct'))}")
    concession_pooled = result["dimensions"].get("concession_pooled", {})
    if concession_pooled:
        print(f"  concession wκ (1–3 scale):    {_fmt(concession_pooled.get('weighted_kappa_quadratic'))}")
    print()
    print(f"Report written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())