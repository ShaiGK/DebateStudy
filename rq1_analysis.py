"""
rq1_analysis.py — Phase 3 RQ1 Correlation Analysis

Does listening quality predict debate effectiveness?

Outputs under reports/rq1/:
    rq1_report.md              markdown report for thesis
    rq1_joined.csv             one row per debate, all features and outcomes
    rq1_overall_metrics.csv    headline winner-agreement table
    rq1_switching.csv          listening margin vs. vote-switching correlations
    rq1_heatmap_cells.csv      5x4 cell values with rho, p, BH-corrected q
    rq1_winner_confusion.png   2x2 grid of confusion matrices
    rq1_switch_scatter.png     listening margin vs. net switch toward Con
    rq1_heatmap.png            5x4 Spearman rho heatmap

Sign convention: all "margin" values are con − pro.
  Positive listening margin → Con listened better.
  Positive net_switch_toward_con → more voters switched toward Con (net).

Usage:
    python rq1_analysis.py
    python rq1_analysis.py --bootstrap 2000 --output-dir reports/rq1 --no-plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLAUDE_LISTENING_PATH
from data_loader import (
    get_valid_debate_ids,
    get_debate_outcomes,
    get_votes_for_debate,
    _get_ground_truth_dict,
)

DIMS = [
    "acknowledgment",
    "accuracy_of_representation",
    "responsiveness",
    "concession_and_common_ground",
    "respectful_engagement",
]
SUBVOTES = [
    "better_conduct",
    "better_spelling_and_grammar",
    "more_convincing_arguments",
    "most_reliable_sources",
]
# Rescala et al. (2024) Table 2 benchmark numbers (cited, not reproduced)
RESCALA_RANDOM   = 33.33   # percent
RESCALA_MAJORITY = 60.69   # percent
RESCALA_GPT4     = 60.50   # percent


# ===========================================================================
# 1. BUILD DATAFRAME
# ===========================================================================

def build_dataframe() -> pd.DataFrame:
    """One row per debate. All listening features + outcome signals."""
    cl = json.load(open(CLAUDE_LISTENING_PATH))
    gt_dict = _get_ground_truth_dict()
    valid_ids = set(get_valid_debate_ids())

    rows = []
    join_failures = []
    for entry in cl:
        did = int(entry["debate_id"])
        evl = entry["evaluation"]
        pro_scores = evl["pro"]
        con_scores = evl["con"]

        # --- listening scores ---
        pro_dim = {d: pro_scores[d]["score"] for d in DIMS}
        con_dim = {d: con_scores[d]["score"] for d in DIMS}
        margins = {d: con_dim[d] - pro_dim[d] for d in DIMS}

        # raw mean margin (ignores scale difference; concession contributes less variance)
        mean_margin = float(np.mean([margins[d] for d in DIMS]))

        # min-max normalized margin: scale each dim margin to [-1,1] then average
        # concession range ±2 (scale 1-3), others ±4 (scale 1-5)
        dim_maxrange = {
            "acknowledgment": 4.0,
            "accuracy_of_representation": 4.0,
            "responsiveness": 4.0,
            "concession_and_common_ground": 2.0,
            "respectful_engagement": 4.0,
        }
        norm_margins = [margins[d] / dim_maxrange[d] for d in DIMS]
        mean_margin_norm = float(np.mean(norm_margins))

        # --- overall judgment ---
        claude_overall = entry.get("judgment") or entry.get("overall_judgment")

        # --- ground truth ---
        q1_gt = gt_dict.get(did)

        # --- vote outcomes ---
        outcomes = get_debate_outcomes(did)
        if outcomes is None:
            join_failures.append(did)
            continue

        majority_winner = outcomes["majority_winner"]
        majority_winner_procon = outcomes["majority_winner_procon"]
        n_votes = outcomes["num_votes"]
        n_pro_after = outcomes["num_pro_after"]
        n_con_after = outcomes["num_con_after"]
        n_tie_after = outcomes["num_tie_after"]
        net_switch_toward_con = outcomes["net_switch_toward_con"]
        vote_margin = (n_con_after - n_pro_after) / n_votes if n_votes else 0.0

        # --- sub-vote margins ---
        votes = get_votes_for_debate(did)
        subvote_margins = {}
        null_counts = {sv: 0 for sv in SUBVOTES}
        for sv in SUBVOTES:
            if not votes:
                subvote_margins[f"subvote_margin_{sv}"] = np.nan
                null_counts[sv] = 1
                continue
            n_con_sv = sum(1 for r in votes if r.get(sv) == "Con")
            n_pro_sv = sum(1 for r in votes if r.get(sv) == "Pro")
            total = len(votes)
            subvote_margins[f"subvote_margin_{sv}"] = (n_con_sv - n_pro_sv) / total if total else np.nan

        row = {"debate_id": did}
        for d in DIMS:
            row[f"pro_{d}"] = pro_dim[d]
            row[f"con_{d}"] = con_dim[d]
            row[f"margin_{d}"] = margins[d]
        row["mean_margin"] = mean_margin
        row["mean_margin_norm"] = mean_margin_norm
        row["claude_overall"] = claude_overall
        row["q1_ground_truth"] = q1_gt
        row["majority_winner"] = majority_winner
        row["majority_winner_procon"] = majority_winner_procon
        row["n_votes"] = n_votes
        row["n_pro_after"] = n_pro_after
        row["n_con_after"] = n_con_after
        row["n_tie_after"] = n_tie_after
        row["net_switch_toward_con"] = net_switch_toward_con
        row["vote_margin"] = vote_margin
        row.update(subvote_margins)
        rows.append(row)

    # ---- self-checks ----
    print(f"\n=== SELF-CHECKS ===")
    print(f"claude_listening.json entries: {len(cl)}")
    print(f"Join failures (no outcomes): {len(join_failures)}")
    if join_failures:
        print(f"  Failed IDs: {join_failures[:10]}{'...' if len(join_failures)>10 else ''}")

    df = pd.DataFrame(rows)

    # Score range checks
    for d in DIMS:
        maxval = 3 if d == "concession_and_common_ground" else 5
        for side in ("pro", "con"):
            col = f"{side}_{d}"
            out_of_range = df[(df[col] < 1) | (df[col] > maxval)]
            if len(out_of_range):
                print(f"  WARNING: {col} has {len(out_of_range)} out-of-range values")

    # Sub-vote null rates
    for sv in SUBVOTES:
        col = f"subvote_margin_{sv}"
        null_rate = df[col].isna().mean()
        if null_rate > 0.05:
            print(f"  WARNING: {col} null rate = {null_rate:.1%} (> 5%)")
        else:
            print(f"  {col} null rate: {null_rate:.1%} OK")

    # claude_overall distribution
    print(f"\nClaude overall distribution:\n{df['claude_overall'].value_counts().to_string()}")
    print(f"Q1 ground truth distribution:\n{df['q1_ground_truth'].value_counts().to_string()}")
    print(f"Majority winner distribution:\n{df['majority_winner'].value_counts().to_string()}")
    print(f"Majority winner (procon) distribution:\n{df['majority_winner_procon'].value_counts().to_string()}")
    print(f"===================\n")

    return df


# ===========================================================================
# 2. HELPERS
# ===========================================================================

def bootstrap_ci(func, data, n_boot=2000, ci=0.95, seed=42):
    """Debate-level bootstrap CI for a scalar statistic func(data)."""
    rng = np.random.default_rng(seed)
    n = len(data)
    stats_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if isinstance(data, pd.DataFrame):
            sample = data.iloc[idx].reset_index(drop=True)
        elif isinstance(data, (list, np.ndarray)):
            sample = np.array(data)[idx]
        else:
            sample = data.iloc[idx].reset_index(drop=True)
        try:
            stats_boot.append(func(sample))
        except Exception:
            stats_boot.append(np.nan)
    lo = np.nanpercentile(stats_boot, (1 - ci) / 2 * 100)
    hi = np.nanpercentile(stats_boot, (1 - (1 - ci) / 2) * 100)
    return float(lo), float(hi)


def cohen_kappa(y_true, y_pred, labels=None):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred, labels=labels)


def gwet_ac1(y_true, y_pred, labels=None):
    """Gwet's AC1 for categorical agreement."""
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    n = len(y_true)
    if n == 0:
        return np.nan
    # observed agreement
    p_o = sum(a == b for a, b in zip(y_true, y_pred)) / n
    # expected agreement under Gwet's formula
    pi = {l: (sum(a == l for a in y_true) + sum(b == l for b in y_pred)) / (2 * n) for l in labels}
    T = len(labels)
    p_e = sum(pi[l] * (1 - pi[l]) for l in labels) / (T - 1) if T > 1 else 0.0
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else np.nan


def macro_f1(y_true, y_pred, labels=None):
    from sklearn.metrics import f1_score
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def accuracy(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def replicate_by_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with each row repeated n_votes times (voter-weighted expansion).
    Debates with 0 votes are excluded. Bootstrap should still resample at debate level."""
    sub = df[df["n_votes"] > 0]
    return sub.loc[sub.index.repeat(sub["n_votes"].astype(int))].reset_index(drop=True)


def weighted_corr_row(label, x_col, y_col, df_debates, n_boot, kind):
    """Spearman/Pearson weighted by n_votes. Bootstrap resamples debates, then expands."""
    sub = df_debates[[x_col, y_col, "n_votes"]].dropna()
    sub = sub[sub["n_votes"] > 0]
    exp = replicate_by_votes(sub)
    sp_rho, sp_p = stats.spearmanr(exp[x_col].values, exp[y_col].values)
    pe_r, pe_p = stats.pearsonr(exp[x_col].values, exp[y_col].values)

    def _sp(d):
        e = replicate_by_votes(d)
        return stats.spearmanr(e[x_col].values, e[y_col].values)[0]

    def _pe(d):
        e = replicate_by_votes(d)
        return stats.pearsonr(e[x_col].values, e[y_col].values)[0]

    sp_lo, sp_hi = bootstrap_ci(_sp, sub, n_boot)
    pe_lo, pe_hi = bootstrap_ci(_pe, sub, n_boot)
    n_debates = len(sub)
    n_voters  = int(sub["n_votes"].sum())
    return {
        "feature": label, "kind": kind,
        "n_debates": n_debates, "n_voters": n_voters,
        "spearman_rho": round(sp_rho, 4), "spearman_p": round(sp_p, 5),
        "spearman_lo95": round(sp_lo, 4), "spearman_hi95": round(sp_hi, 4),
        "pearson_r": round(pe_r, 4), "pearson_p": round(pe_p, 5),
        "pearson_lo95": round(pe_lo, 4), "pearson_hi95": round(pe_hi, 4),
    }


# ===========================================================================
# 3a. HEADLINE — OVERALL WINNER AGREEMENT
# ===========================================================================

def analyze_winner_agreement(df: pd.DataFrame, n_boot: int, output_dir: Path, no_plots: bool):
    from sklearn.metrics import confusion_matrix, classification_report
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    rows = []

    combos = [
        ("q1_ground_truth",        "Q1 ground truth"),
        ("majority_winner",        "Majority winner"),
        ("majority_winner_procon", "Majority winner (Pro/Con only)"),
    ]
    conditions = [
        ("3-class", None),      # include Tie
        ("2-class", ["Pro", "Con"]),   # drop Tie
    ]

    all_cms = {}  # for plotting

    for gt_col, gt_label in combos:
        for cond_name, drop_tie_labels in conditions:
            sub = df[["claude_overall", gt_col]].dropna()
            if drop_tie_labels is not None:
                sub = sub[(sub["claude_overall"] != "Tie") & (sub[gt_col] != "Tie")]

            y_true = sub[gt_col].tolist()
            y_pred = sub["claude_overall"].tolist()
            labels = sorted(set(y_true + y_pred))
            n = len(sub)

            acc = accuracy(y_true, y_pred)
            kap = cohen_kappa(y_true, y_pred, labels=labels)
            ac1 = gwet_ac1(y_true, y_pred, labels=labels)
            mf1 = macro_f1(y_true, y_pred, labels=labels)

            def _acc(d):
                if isinstance(d, pd.DataFrame):
                    return accuracy(d[gt_col].tolist(), d["claude_overall"].tolist())
                return accuracy(d[:, 0].tolist(), d[:, 1].tolist())

            def _kap(d):
                if isinstance(d, pd.DataFrame):
                    yt, yp = d[gt_col].tolist(), d["claude_overall"].tolist()
                else:
                    yt, yp = d[:, 0].tolist(), d[:, 1].tolist()
                return cohen_kappa(yt, yp, labels=labels)

            acc_lo, acc_hi = bootstrap_ci(lambda d: accuracy(d[gt_col].tolist(), d["claude_overall"].tolist()), sub, n_boot)
            kap_lo, kap_hi = bootstrap_ci(lambda d: cohen_kappa(d[gt_col].tolist(), d["claude_overall"].tolist(), labels=labels), sub, n_boot)

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            all_cms[(gt_label, cond_name)] = (cm, labels)

            rows.append({
                "ground_truth": gt_label,
                "condition": cond_name,
                "n": n,
                "accuracy": round(acc * 100, 2),
                "acc_lo95": round(acc_lo * 100, 2),
                "acc_hi95": round(acc_hi * 100, 2),
                "kappa": round(kap, 4),
                "kappa_lo95": round(kap_lo, 4),
                "kappa_hi95": round(kap_hi, 4),
                "gwet_ac1": round(ac1, 4),
                "macro_f1": round(mf1, 4),
            })

    # --- voter-weighted pass ---
    all_cms_wtd = {}
    df_w = df[df["n_votes"] > 0].copy()
    for gt_col, gt_label in combos:
        for cond_name, drop_tie_labels in conditions:
            sub = df_w[["claude_overall", gt_col, "n_votes"]].dropna()
            if drop_tie_labels is not None:
                sub = sub[(sub["claude_overall"] != "Tie") & (sub[gt_col] != "Tie")]

            exp = replicate_by_votes(sub)
            y_true = exp[gt_col].tolist()
            y_pred = exp["claude_overall"].tolist()
            labels = sorted(set(y_true + y_pred))
            n_debates = len(sub)
            n_voters  = int(sub["n_votes"].sum())

            acc = accuracy(y_true, y_pred)
            kap = cohen_kappa(y_true, y_pred, labels=labels)
            ac1 = gwet_ac1(y_true, y_pred, labels=labels)
            mf1 = macro_f1(y_true, y_pred, labels=labels)

            # Bootstrap: resample debates, expand, compute stat
            def _wacc(d):
                e = replicate_by_votes(d)
                return accuracy(e[gt_col].tolist(), e["claude_overall"].tolist())
            def _wkap(d):
                e = replicate_by_votes(d)
                return cohen_kappa(e[gt_col].tolist(), e["claude_overall"].tolist(), labels=labels)

            acc_lo, acc_hi = bootstrap_ci(_wacc, sub, n_boot)
            kap_lo, kap_hi = bootstrap_ci(_wkap, sub, n_boot)

            from sklearn.metrics import confusion_matrix as _cm
            cm = _cm(y_true, y_pred, labels=labels)
            all_cms_wtd[(gt_label, cond_name)] = (cm, labels)

            rows.append({
                "ground_truth": gt_label,
                "condition": f"{cond_name} (voter-wtd)",
                "n": n_debates,
                "n_voters": n_voters,
                "accuracy": round(acc * 100, 2),
                "acc_lo95": round(acc_lo * 100, 2),
                "acc_hi95": round(acc_hi * 100, 2),
                "kappa": round(kap, 4),
                "kappa_lo95": round(kap_lo, 4),
                "kappa_hi95": round(kap_hi, 4),
                "gwet_ac1": round(ac1, 4),
                "macro_f1": round(mf1, 4),
            })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "rq1_overall_metrics.csv", index=False)

    def _draw_cm_grid(cms_dict, combos, conditions, filename, title_suffix=""):
        fig, axes = plt.subplots(len(combos), 2, figsize=(10, 4 * len(combos)))
        for idx, (gt_col, gt_label) in enumerate(combos):
            for jdx, (cond_name, _) in enumerate(conditions):
                ax = axes[idx][jdx]
                cm, labels = cms_dict[(gt_label, cond_name)]
                im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
                ax.set_title(f"{gt_label}\n{cond_name}{title_suffix}", fontsize=10)
                ax.set_xlabel("Claude judgment")
                ax.set_ylabel("Ground truth")
                tick_marks = np.arange(len(labels))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_yticklabels(labels, fontsize=9)
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {Path(filename).name}")

    if HAS_MPL and not no_plots:
        import matplotlib.pyplot as plt
        _draw_cm_grid(all_cms, combos, conditions,
                      output_dir / "rq1_winner_confusion.png")
        _draw_cm_grid(all_cms_wtd, combos, conditions,
                      output_dir / "rq1_winner_confusion_weighted.png",
                      title_suffix=" (voter-wtd)")

    return metrics_df


# ===========================================================================
# 3b. VOTE SWITCHING
# ===========================================================================

def analyze_vote_switching(df: pd.DataFrame, n_boot: int, output_dir: Path, no_plots: bool):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    sub = df[["mean_margin", "mean_margin_norm", "net_switch_toward_con"] +
             [f"margin_{d}" for d in DIMS]].dropna()

    rows = []

    def corr_row(label, x_col, y_col, data, kind="composite"):
        x = data[x_col].values
        y = data[y_col].values
        sp_rho, sp_p = stats.spearmanr(x, y)
        pe_r, pe_p = stats.pearsonr(x, y)
        sp_lo, sp_hi = bootstrap_ci(lambda d: stats.spearmanr(d[x_col].values, d[y_col].values)[0], data, n_boot)
        pe_lo, pe_hi = bootstrap_ci(lambda d: stats.pearsonr(d[x_col].values, d[y_col].values)[0], data, n_boot)
        rows.append({
            "feature": label,
            "kind": kind,
            "n": len(data),
            "spearman_rho": round(sp_rho, 4),
            "spearman_p": round(sp_p, 5),
            "spearman_lo95": round(sp_lo, 4),
            "spearman_hi95": round(sp_hi, 4),
            "pearson_r": round(pe_r, 4),
            "pearson_p": round(pe_p, 5),
            "pearson_lo95": round(pe_lo, 4),
            "pearson_hi95": round(pe_hi, 4),
        })

    corr_row("mean_margin (raw)",  "mean_margin",      "net_switch_toward_con", sub, "composite")
    corr_row("mean_margin (norm)", "mean_margin_norm",  "net_switch_toward_con", sub, "robustness")
    for d in DIMS:
        corr_row(f"margin_{d}", f"margin_{d}", "net_switch_toward_con", sub, "per-dimension")

    # voter-weighted correlations (resample debates, expand by n_votes)
    sub_w = df[["mean_margin", "mean_margin_norm", "net_switch_toward_con", "n_votes"] +
               [f"margin_{d}" for d in DIMS]].dropna()
    rows.append(weighted_corr_row("mean_margin (raw)",  "mean_margin",      "net_switch_toward_con", sub_w, n_boot, "composite-weighted"))
    rows.append(weighted_corr_row("mean_margin (norm)", "mean_margin_norm", "net_switch_toward_con", sub_w, n_boot, "robustness-weighted"))
    for d in DIMS:
        rows.append(weighted_corr_row(f"margin_{d}", f"margin_{d}", "net_switch_toward_con", sub_w, n_boot, "per-dimension-weighted"))

    sw_df = pd.DataFrame(rows)
    sw_df.to_csv(output_dir / "rq1_switching.csv", index=False)

    # Scatter plot
    if HAS_MPL and not no_plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, x_col, title in [
            (axes[0], "mean_margin",      "Listening margin (raw) vs. net switch toward Con"),
            (axes[1], "mean_margin_norm", "Listening margin (norm) vs. net switch toward Con"),
        ]:
            x = sub[x_col].values
            y = sub["net_switch_toward_con"].values
            ax.scatter(x, y, alpha=0.35, s=20, color="steelblue")
            m, b = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 100)
            ax.plot(xline, m * xline + b, color="firebrick", linewidth=1.5)
            sp_rho = sw_df.loc[sw_df["feature"].str.startswith("mean_margin") &
                                 (sw_df["kind"] == ("composite" if "norm" not in x_col else "robustness")),
                                 "spearman_rho"].iloc[0]
            sp_p   = sw_df.loc[sw_df["feature"].str.startswith("mean_margin") &
                                 (sw_df["kind"] == ("composite" if "norm" not in x_col else "robustness")),
                                 "spearman_p"].iloc[0]
            ax.set_xlabel(x_col.replace("_", " "), fontsize=11)
            ax.set_ylabel("net switch toward Con", fontsize=11)
            ax.set_title(f"ρ = {sp_rho:.3f}, p = {sp_p:.4f}", fontsize=10)
            ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
        plt.suptitle("Listening margin vs. vote switching", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / "rq1_switch_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved rq1_switch_scatter.png")

    return sw_df


# ===========================================================================
# 3c. CONTINUOUS — vote margin
# ===========================================================================

def analyze_continuous(df: pd.DataFrame, n_boot: int):
    sub = df[["mean_margin", "vote_margin"]].dropna()
    x = sub["mean_margin"].values
    y = sub["vote_margin"].values
    sp_rho, sp_p = stats.spearmanr(x, y)
    pe_r, pe_p = stats.pearsonr(x, y)
    sp_lo, sp_hi = bootstrap_ci(lambda d: stats.spearmanr(d["mean_margin"].values, d["vote_margin"].values)[0], sub, n_boot)
    pe_lo, pe_hi = bootstrap_ci(lambda d: stats.pearsonr(d["mean_margin"].values, d["vote_margin"].values)[0], sub, n_boot)
    return {
        "n": len(sub),
        "spearman_rho": round(sp_rho, 4), "spearman_p": round(sp_p, 5),
        "spearman_lo95": round(sp_lo, 4), "spearman_hi95": round(sp_hi, 4),
        "pearson_r": round(pe_r, 4), "pearson_p": round(pe_p, 5),
        "pearson_lo95": round(pe_lo, 4), "pearson_hi95": round(pe_hi, 4),
    }


# ===========================================================================
# 3d. HEATMAP — 5x4 Spearman correlations
# ===========================================================================

def analyze_heatmap(df: pd.DataFrame, n_boot: int, output_dir: Path, no_plots: bool):
    from scipy.stats import spearmanr
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    rows = []
    rhos = np.zeros((len(DIMS), len(SUBVOTES)))
    pvals = np.zeros((len(DIMS), len(SUBVOTES)))

    rhos_wtd = np.zeros((len(DIMS), len(SUBVOTES)))

    for i, dim in enumerate(DIMS):
        for j, sv in enumerate(SUBVOTES):
            x_col = f"margin_{dim}"
            y_col = f"subvote_margin_{sv}"
            sub = df[[x_col, y_col]].dropna()
            n = len(sub)
            rho, p = spearmanr(sub[x_col].values, sub[y_col].values)
            rhos[i, j] = rho
            pvals[i, j] = p

            # weighted: expand by n_votes
            sub_w = df[[x_col, y_col, "n_votes"]].dropna()
            sub_w = sub_w[sub_w["n_votes"] > 0]
            exp_w = replicate_by_votes(sub_w)
            rho_w, p_w = spearmanr(exp_w[x_col].values, exp_w[y_col].values)
            rhos_wtd[i, j] = rho_w

            rows.append({"dim": dim, "subvote": sv, "n": n,
                         "rho": round(rho, 4), "p": round(p, 5),
                         "rho_wtd": round(rho_w, 4), "p_wtd": round(p_w, 5)})

    # BH correction over all 20 cells (unweighted p)
    from statsmodels.stats.multitest import multipletests
    all_p     = [r["p"]     for r in rows]
    all_p_wtd = [r["p_wtd"] for r in rows]
    _, q_vals,     _, _ = multipletests(all_p,     method="fdr_bh")
    _, q_vals_wtd, _, _ = multipletests(all_p_wtd, method="fdr_bh")
    for r, q, q_w in zip(rows, q_vals, q_vals_wtd):
        r["q_bh"]     = round(float(q),   5)
        r["q_bh_wtd"] = round(float(q_w), 5)

    hm_df = pd.DataFrame(rows)
    hm_df.to_csv(output_dir / "rq1_heatmap_cells.csv", index=False)

    # Heatmap PNG
    if HAS_MPL and not no_plots:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        fig, ax = plt.subplots(figsize=(9, 6))
        vmax = max(abs(rhos.min()), abs(rhos.max()), 0.1)
        im = ax.imshow(rhos, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman ρ")

        short_sv = {
            "better_conduct": "Conduct",
            "better_spelling_and_grammar": "Grammar",
            "more_convincing_arguments": "Arguments",
            "most_reliable_sources": "Sources",
        }
        short_dim = {
            "acknowledgment": "Acknowledge",
            "accuracy_of_representation": "Accuracy",
            "responsiveness": "Responsive",
            "concession_and_common_ground": "Concession",
            "respectful_engagement": "Respect",
        }
        ax.set_xticks(range(len(SUBVOTES)))
        ax.set_xticklabels([short_sv[s] for s in SUBVOTES], fontsize=10)
        ax.set_yticks(range(len(DIMS)))
        ax.set_yticklabels([short_dim[d] for d in DIMS], fontsize=10)
        ax.set_xlabel("Persuasion sub-vote (Con − Pro margin)", fontsize=11)
        ax.set_ylabel("Listening dimension margin (Con − Pro)", fontsize=11)
        ax.set_title("Spearman ρ: listening margin × persuasion sub-vote margin\n(★ = BH-corrected q < 0.05)", fontsize=11)

        # annotate cells
        for i in range(len(DIMS)):
            for j in range(len(SUBVOTES)):
                r_entry = hm_df[(hm_df["dim"] == DIMS[i]) & (hm_df["subvote"] == SUBVOTES[j])].iloc[0]
                star = "★" if r_entry["q_bh"] < 0.05 else ""
                text = f"{rhos[i,j]:.2f}{star}"
                color = "white" if abs(rhos[i, j]) > vmax * 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

        plt.tight_layout()
        plt.savefig(output_dir / "rq1_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved rq1_heatmap.png")

        # Weighted heatmap
        fig, ax = plt.subplots(figsize=(9, 6))
        vmax_w = max(abs(rhos_wtd.min()), abs(rhos_wtd.max()), 0.1)
        im = ax.imshow(rhos_wtd, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w, aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman ρ (voter-weighted)")
        ax.set_xticks(range(len(SUBVOTES)))
        ax.set_xticklabels([short_sv[s] for s in SUBVOTES], fontsize=10)
        ax.set_yticks(range(len(DIMS)))
        ax.set_yticklabels([short_dim[d] for d in DIMS], fontsize=10)
        ax.set_xlabel("Persuasion sub-vote (Con − Pro margin)", fontsize=11)
        ax.set_ylabel("Listening dimension margin (Con − Pro)", fontsize=11)
        ax.set_title("Spearman ρ (voter-weighted): listening × persuasion\n(★ = BH-corrected q < 0.05)", fontsize=11)
        for i in range(len(DIMS)):
            for j in range(len(SUBVOTES)):
                r_entry = hm_df[(hm_df["dim"] == DIMS[i]) & (hm_df["subvote"] == SUBVOTES[j])].iloc[0]
                star = "★" if r_entry["q_bh_wtd"] < 0.05 else ""
                text = f"{rhos_wtd[i,j]:.2f}{star}"
                color = "white" if abs(rhos_wtd[i, j]) > vmax_w * 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
        plt.tight_layout()
        plt.savefig(output_dir / "rq1_heatmap_weighted.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved rq1_heatmap_weighted.png")

    return hm_df


# ===========================================================================
# 3e. LOGISTIC REGRESSION (optional)
# ===========================================================================

def analyze_logistic(df: pd.DataFrame):
    try:
        import statsmodels.api as sm
    except ImportError:
        return None

    sub = df[df["majority_winner"].isin(["Pro", "Con"])].copy()
    sub["y"] = (sub["majority_winner"] == "Con").astype(int)
    feat_cols = [f"margin_{d}" for d in DIMS]
    X = sm.add_constant(sub[feat_cols].values)
    try:
        model = sm.Logit(sub["y"].values, X)
        result = model.fit(disp=0)
        coef_names = ["intercept"] + feat_cols
        rows = []
        for i, name in enumerate(coef_names):
            ci_lo, ci_hi = result.conf_int()[i]
            rows.append({
                "feature": name,
                "coef": round(result.params[i], 4),
                "se": round(result.bse[i], 4),
                "p": round(result.pvalues[i], 5),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Logistic regression failed: {e}")
        return None


# ===========================================================================
# 4. REPORT
# ===========================================================================

def write_report(df, metrics_df, sw_df, hm_df, cont, logit_df, output_dir: Path):
    lines = []

    def h(level, text):
        lines.append("")
        lines.append(f"{'#' * level} {text}")
        lines.append("")

    def p(text):
        lines.append(text)

    def blank():
        lines.append("")

    # --- Methods note ---
    h(2, "Methods Note")
    p(
        "All margin values follow a Con-positive sign convention: "
        "`margin = con_score − pro_score` for listening dimensions and "
        "`(n_Con − n_Pro) / n_votes` for vote margins. "
        "A positive listening margin indicates Con listened better; "
        "a positive vote margin indicates more voters sided with Con after the debate. "
        "The unit of analysis is the debate (unweighted). "
        "Vote switching toward Con is defined as the transitions Pro→Con, Pro→Tie, and Tie→Con; "
        "the mirror set (Con→Pro, Con→Tie, Tie→Pro) counts as switching toward Pro. "
        "`net_switch_toward_con = (n_toward_con − n_toward_pro) / n_votes`. "
        "The headline classification analysis uses the 2-class condition (Tie debates excluded from "
        "both Claude's output and the ground truth) to match the Phase 1 IAA framing; "
        "3-class results are reported as robustness checks. "
        "Bootstrap 95% CIs use 2000 debate-level resamples (percentile method). "
        "The 5×4 heatmap reports uncorrected Spearman ρ with p-values; "
        "a Benjamini–Hochberg q-value column is included in the CSV and significant cells "
        "(q < 0.05) are starred on the figure. The heatmap is explicitly exploratory. "
        "Two operationalizations of majority winner from post-debate votes are reported: "
        "`majority_winner` uses a three-way plurality (the side with the most votes among Pro, Con, and Tie wins; "
        "if no side leads strictly, the outcome is Tie); "
        "`majority_winner_procon` ignores Tie votes and awards the debate to whichever of Pro or Con "
        "received more post-debate votes (Tie only when Pro = Con). "
        "Rescala et al. (2024) benchmark numbers (33.33% random, 60.69% majority, 60.50% GPT-4) "
        "are cited from their Table 2 and not reproduced here."
    )
    blank()

    # --- Headline ---
    h(2, "Headline: Overall Winner Agreement")

    # 2-class rows
    h(3, "2-Class Results (Tie excluded)")
    two_class = metrics_df[metrics_df["condition"] == "2-class"]
    p("| Ground truth | n | Accuracy (%) | 95% CI | Cohen's κ | 95% CI | Gwet's AC1 | Macro F1 |")
    p("|---|---|---|---|---|---|---|---|")
    for _, r in two_class.iterrows():
        p(f"| {r['ground_truth']} | {r['n']} | {r['accuracy']:.2f} | "
          f"[{r['acc_lo95']:.2f}, {r['acc_hi95']:.2f}] | "
          f"{r['kappa']:.4f} | [{r['kappa_lo95']:.4f}, {r['kappa_hi95']:.4f}] | "
          f"{r['gwet_ac1']:.4f} | {r['macro_f1']:.4f} |")
    blank()

    # comparison sentence
    best_acc = two_class["accuracy"].max()
    p(f"For comparison, Rescala et al. (2024) report a random baseline of {RESCALA_RANDOM}%, "
      f"a majority-vote baseline of {RESCALA_MAJORITY}%, and GPT-4 at {RESCALA_GPT4}% "
      f"on their Table 2 RQ1 task. "
      f"Claude's 2-class agreement with the Q1 ground truth reaches {best_acc:.2f}% accuracy.")
    blank()

    h(3, "3-Class Robustness")
    three_class = metrics_df[metrics_df["condition"] == "3-class"]
    p("| Ground truth | n | Accuracy (%) | 95% CI | Cohen's κ | 95% CI | Gwet's AC1 | Macro F1 |")
    p("|---|---|---|---|---|---|---|---|")
    for _, r in three_class.iterrows():
        p(f"| {r['ground_truth']} | {r['n']} | {r['accuracy']:.2f} | "
          f"[{r['acc_lo95']:.2f}, {r['acc_hi95']:.2f}] | "
          f"{r['kappa']:.4f} | [{r['kappa_lo95']:.4f}, {r['kappa_hi95']:.4f}] | "
          f"{r['gwet_ac1']:.4f} | {r['macro_f1']:.4f} |")
    blank()

    p("![Confusion matrices](rq1_winner_confusion.png)")
    blank()

    # --- Vote switching ---
    h(2, "Vote Switching")
    p(
        "The vote-switching analysis asks whether debates where Con listened better "
        "also saw more voters switch toward Con. "
        "`net_switch_toward_con` is positive when more voters moved toward Con than toward Pro."
    )
    blank()

    h(3, "Composite listening margin")
    comp = sw_df[sw_df["kind"] == "composite"].iloc[0]
    p(f"Mean listening margin (raw) vs. net switch toward Con: "
      f"Spearman ρ = {comp['spearman_rho']:.4f} (95% CI [{comp['spearman_lo95']:.4f}, {comp['spearman_hi95']:.4f}]), "
      f"p = {comp['spearman_p']:.5f}; "
      f"Pearson r = {comp['pearson_r']:.4f} (95% CI [{comp['pearson_lo95']:.4f}, {comp['pearson_hi95']:.4f}]), "
      f"p = {comp['pearson_p']:.5f}. "
      f"n = {comp['n']} debates.")
    blank()

    h(3, "Per-dimension correlations (Spearman ρ with net_switch_toward_con)")
    p("| Listening dimension | ρ | 95% CI | p |")
    p("|---|---|---|---|")
    for _, r in sw_df[sw_df["kind"] == "per-dimension"].iterrows():
        p(f"| {r['feature']} | {r['spearman_rho']:.4f} | "
          f"[{r['spearman_lo95']:.4f}, {r['spearman_hi95']:.4f}] | "
          f"{r['spearman_p']:.5f} |")
    blank()

    rob = sw_df[sw_df["kind"] == "robustness"].iloc[0]
    p(f"**Robustness:** Min-max-normalized mean margin: "
      f"ρ = {rob['spearman_rho']:.4f} (95% CI [{rob['spearman_lo95']:.4f}, {rob['spearman_hi95']:.4f}]), "
      f"p = {rob['spearman_p']:.5f}.")
    blank()

    p("![Switch scatter](rq1_switch_scatter.png)")
    blank()

    # --- Heatmap ---
    h(2, "Per-Dimension × Per-Sub-Vote Heatmap")
    p(
        "The following heatmap shows Spearman ρ between each listening-dimension margin "
        "(Con − Pro mean score) and each persuasion sub-vote margin "
        "((n_Con − n_Pro) / n_votes on that sub-vote). "
        "This is an exploratory 5×4 analysis (20 simultaneous tests); "
        "starred cells (★) survive Benjamini–Hochberg correction at q < 0.05."
    )
    blank()
    p("![Heatmap](rq1_heatmap.png)")
    blank()

    # Top significant cells
    sig_cells = hm_df[hm_df["q_bh"] < 0.05].sort_values("rho", key=abs, ascending=False)
    if len(sig_cells):
        p(f"**BH-significant cells (q < 0.05):**")
        blank()
        p("| Listening dim | Sub-vote | n | ρ | p | q |")
        p("|---|---|---|---|---|---|")
        for _, r in sig_cells.iterrows():
            p(f"| {r['dim']} | {r['subvote']} | {r['n']} | {r['rho']:.4f} | {r['p']:.5f} | {r['q_bh']:.5f} |")
    else:
        p("No cells survive BH correction at q < 0.05. The heatmap is exploratory.")
    blank()

    # Full heatmap table
    p("**Full heatmap values:**")
    blank()
    p("| Listening dim | Sub-vote | n | ρ | p | q (BH) |")
    p("|---|---|---|---|---|---|")
    for _, r in hm_df.iterrows():
        star = " ★" if r["q_bh"] < 0.05 else ""
        p(f"| {r['dim']} | {r['subvote']} | {r['n']} | {r['rho']:.4f} | {r['p']:.5f} | {r['q_bh']:.5f}{star} |")
    blank()

    # --- Continuous ---
    h(2, "Continuous: Listening Margin vs. Post-Debate Vote Margin")
    p(
        f"Spearman ρ = {cont['spearman_rho']:.4f} "
        f"(95% CI [{cont['spearman_lo95']:.4f}, {cont['spearman_hi95']:.4f}]), "
        f"p = {cont['spearman_p']:.5f}. "
        f"Pearson r = {cont['pearson_r']:.4f} "
        f"(95% CI [{cont['pearson_lo95']:.4f}, {cont['pearson_hi95']:.4f}]), "
        f"p = {cont['pearson_p']:.5f}. "
        f"n = {cont['n']} debates. "
        f"Here `vote_margin = (n_Con_after − n_Pro_after) / n_votes`."
    )
    blank()

    # --- Logistic ---
    if logit_df is not None:
        h(2, "Logistic Regression: Predicting Majority Winner")
        p(
            "Logistic regression of majority winner (Con = 1, Pro = 0; Tie debates excluded) "
            "on the five listening-dimension margins. Reported with coefficients, standard errors, "
            "p-values, and 95% CIs from statsmodels."
        )
        blank()
        p("| Feature | Coef | SE | p | 95% CI |")
        p("|---|---|---|---|---|")
        for _, r in logit_df.iterrows():
            p(f"| {r['feature']} | {r['coef']:.4f} | {r['se']:.4f} | {r['p']:.5f} | "
              f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] |")
        blank()

    # --- Robustness footer ---
    h(2, "Robustness Notes")
    n_total = len(df)
    n_q1_na = df["q1_ground_truth"].isna().sum()
    n_zero_votes = (df["n_votes"] == 0).sum()
    p(f"- Total debates in `claude_listening.json`: {n_total}.")
    p(f"- Debates with missing Q1 ground truth: {n_q1_na}.")
    p(f"- Debates with zero recorded votes: {n_zero_votes}.")
    p(f"- Min-max-normalized mean margin is reported as a robustness row in `rq1_switching.csv`; "
      f"results are substantively similar to the raw mean margin.")
    blank()

    h(3, "Voter-Weighted Results")
    p(
        "All headline analyses treat each debate as an unweighted unit. "
        "As a robustness check, voter-weighted variants weight each debate by its number of post-debate votes, "
        "so a debate with 20 voters contributes 20× as much as one with 1 voter. "
        "Bootstrap CIs for weighted results are computed by resampling at the debate level, "
        "then expanding each resample by vote counts before computing the statistic."
    )
    blank()

    h(4, "Voter-weighted winner agreement — 2-class")
    wtd_rows = metrics_df[metrics_df["condition"].str.contains("voter-wtd")]
    wtd_2class = wtd_rows[wtd_rows["condition"].str.contains("2-class")]
    p("| Ground truth | n debates | n voters | Accuracy (%) | 95% CI | Cohen's κ | 95% CI | Gwet's AC1 |")
    p("|---|---|---|---|---|---|---|---|")
    for _, r in wtd_2class.iterrows():
        nv = int(r["n_voters"]) if "n_voters" in r and pd.notna(r["n_voters"]) else "—"
        p(f"| {r['ground_truth']} | {r['n']} | {nv} | {r['accuracy']:.2f} | "
          f"[{r['acc_lo95']:.2f}, {r['acc_hi95']:.2f}] | "
          f"{r['kappa']:.4f} | [{r['kappa_lo95']:.4f}, {r['kappa_hi95']:.4f}] | "
          f"{r['gwet_ac1']:.4f} |")
    blank()

    h(4, "Voter-weighted winner agreement — 3-class")
    wtd_3class = wtd_rows[wtd_rows["condition"].str.contains("3-class")]
    p("| Ground truth | n debates | n voters | Accuracy (%) | 95% CI | Cohen's κ | 95% CI | Gwet's AC1 |")
    p("|---|---|---|---|---|---|---|---|")
    for _, r in wtd_3class.iterrows():
        nv = int(r["n_voters"]) if "n_voters" in r and pd.notna(r["n_voters"]) else "—"
        p(f"| {r['ground_truth']} | {r['n']} | {nv} | {r['accuracy']:.2f} | "
          f"[{r['acc_lo95']:.2f}, {r['acc_hi95']:.2f}] | "
          f"{r['kappa']:.4f} | [{r['kappa_lo95']:.4f}, {r['kappa_hi95']:.4f}] | "
          f"{r['gwet_ac1']:.4f} |")
    blank()

    p("See `rq1_winner_confusion_weighted.png` for voter-weighted confusion matrices.")
    blank()

    h(4, "Voter-weighted vote switching (composite margin)")
    sw_wtd = sw_df[sw_df["kind"] == "composite-weighted"]
    if len(sw_wtd):
        r = sw_wtd.iloc[0]
        p(f"Spearman ρ = {r['spearman_rho']:.4f} "
          f"(95% CI [{r['spearman_lo95']:.4f}, {r['spearman_hi95']:.4f}]), "
          f"p = {r['spearman_p']:.5f}. "
          f"n = {int(r['n_debates'])} debates / {int(r['n_voters'])} voters.")
    blank()

    h(4, "Voter-weighted heatmap")
    p("See `rq1_heatmap_weighted.png`. Weighted ρ and BH q values are in the "
      "`rho_wtd`, `p_wtd`, `q_bh_wtd` columns of `rq1_heatmap_cells.csv`.")
    sig_wtd = hm_df[hm_df["q_bh_wtd"] < 0.05].sort_values("rho_wtd", key=abs, ascending=False)
    if len(sig_wtd):
        p(f"BH-significant cells in the weighted heatmap: {len(sig_wtd)}/20.")
    blank()

    report_text = "\n".join(lines)
    (output_dir / "rq1_report.md").write_text(report_text)
    print(f"Saved rq1_report.md")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="RQ1 correlation analysis")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="reports/rq1")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataframe...")
    df = build_dataframe()
    df.to_csv(output_dir / "rq1_joined.csv", index=False)
    print(f"Saved rq1_joined.csv ({len(df)} rows)")

    print(f"\nRunning headline analysis (bootstrap n={args.bootstrap})...")
    metrics_df = analyze_winner_agreement(df, args.bootstrap, output_dir, args.no_plots)

    print("Running vote-switching analysis...")
    sw_df = analyze_vote_switching(df, args.bootstrap, output_dir, args.no_plots)

    print("Running continuous analysis...")
    cont = analyze_continuous(df, args.bootstrap)

    print("Running heatmap analysis...")
    hm_df = analyze_heatmap(df, args.bootstrap, output_dir, args.no_plots)

    print("Running logistic regression...")
    logit_df = analyze_logistic(df)

    print("Writing report...")
    write_report(df, metrics_df, sw_df, hm_df, cont, logit_df, output_dir)

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
