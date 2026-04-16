"""
rq1_analysis.py — Phase 3 RQ1 Correlation Analysis

Does listening quality predict debate effectiveness?

Outputs under reports/rq1/:
    rq1_report.md                  markdown report for thesis
    rq1_joined.csv                 one row per debate, all features and outcomes
    rq1_overall_metrics.csv        headline winner-agreement table
    rq1_switching.csv              listening margin vs. vote-switching correlations
    rq1_switchers_conditional.csv  voter-level switcher table (conditional analysis)
    rq1_heatmap_cells.csv          5x5 cell values with rho, p, BH-corrected q
    rq1_dim_gt_correlations.csv    per-dimension rho vs 3 binarized ground truths
    rq1_classifier.csv             cross-validated logistic classifier summary
    rq1_winner_confusion.png       2x2 grid of confusion matrices
    rq1_switch_scatter.png         listening margin vs. net switch toward Con
    rq1_switch_confusion.png       confusion matrix: Claude judgment × switch direction
    rq1_heatmap.png                5x5 Spearman rho heatmap
    rq1_dim_gt_barchart.png        bar chart: per-dim rho vs 3 ground truths

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
# 5th heatmap column: continuous post-debate vote margin
HEATMAP_COLS = SUBVOTES + ["vote_margin"]

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
# 3b-conditional. SWITCHERS ONLY — does switch direction match Claude's judgment?
# ===========================================================================

def analyze_switchers_conditional(df: pd.DataFrame, n_boot: int, output_dir: Path, no_plots: bool):
    """
    Among voters who actually switched their vote, does the direction of the switch
    (toward Pro or toward Con) agree with Claude's overall better-listener judgment?

    This is a sharper test than the correlation across all debates: it asks whether
    the side Claude identified as the better listener is the side that persuaded
    the voters who were persuadable enough to change their mind.

    Unit of analysis: individual switch events (voter × debate).
    Bootstrap resamples at the DEBATE level to respect within-debate clustering.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    # Build voter-level switcher table joined with Claude's debate-level judgment
    switcher_rows = []
    for _, row in df.iterrows():
        did = int(row["debate_id"])
        claude_j = row["claude_overall"]
        votes = get_votes_for_debate(did)
        for v in votes:
            ab = v.get("agreed_before")
            aa = v.get("agreed_after")
            if ab is None or aa is None or ab == aa:
                continue  # stable voter — skip
            toward_con = (
                (ab == "Pro" and aa == "Con") or
                (ab == "Pro" and aa == "Tie") or
                (ab == "Tie" and aa == "Con")
            )
            toward_pro = (
                (ab == "Con" and aa == "Pro") or
                (ab == "Con" and aa == "Tie") or
                (ab == "Tie" and aa == "Pro")
            )
            if not toward_con and not toward_pro:
                continue
            switcher_rows.append({
                "debate_id": did,
                "agreed_before": ab,
                "agreed_after": aa,
                "switch_direction": "Con" if toward_con else "Pro",
                "claude_judgment": claude_j,
            })

    sw_voter_df = pd.DataFrame(switcher_rows)
    sw_voter_df.to_csv(output_dir / "rq1_switchers_conditional.csv", index=False)
    print(f"Saved rq1_switchers_conditional.csv ({len(sw_voter_df)} switcher events)")

    n_total_switchers = len(sw_voter_df)
    n_debates_with_switches = sw_voter_df["debate_id"].nunique()

    # ---- headline: all directional switches, Tie claude judgments excluded ----
    sub = sw_voter_df[sw_voter_df["claude_judgment"].isin(["Pro", "Con"])].copy()
    labels = ["Con", "Pro"]

    y_true = sub["switch_direction"].tolist()   # what voters actually did
    y_pred = sub["claude_judgment"].tolist()     # what Claude predicted

    n_2class = len(sub)
    n_debates_2class = sub["debate_id"].nunique()

    acc = accuracy(y_true, y_pred)
    kap = cohen_kappa(y_true, y_pred, labels=labels)
    ac1 = gwet_ac1(y_true, y_pred, labels=labels)
    mf1 = macro_f1(y_true, y_pred, labels=labels)

    # Bootstrap at debate level (switchers within a debate share the same claude_judgment
    # so are not independent; resample debates, then pool their switchers)
    debate_ids_arr = sub["debate_id"].unique()
    debate_switchers_lookup = {
        did: list(zip(g["switch_direction"], g["claude_judgment"]))
        for did, g in sub.groupby("debate_id")
    }

    def _pool(sampled_ids):
        yt, yp = [], []
        for did in sampled_ids:
            for sd, cj in debate_switchers_lookup.get(did, []):
                yt.append(sd)
                yp.append(cj)
        return yt, yp

    rng = np.random.default_rng(42)
    n_d = len(debate_ids_arr)
    acc_boots, kap_boots = [], []
    for _ in range(n_boot):
        sampled = debate_ids_arr[rng.integers(0, n_d, size=n_d)]
        yt, yp = _pool(sampled)
        if len(yt) == 0:
            acc_boots.append(np.nan)
            kap_boots.append(np.nan)
            continue
        acc_boots.append(accuracy(yt, yp))
        try:
            kap_boots.append(cohen_kappa(yt, yp, labels=labels))
        except Exception:
            kap_boots.append(np.nan)

    acc_lo = float(np.nanpercentile(acc_boots, 2.5))
    acc_hi = float(np.nanpercentile(acc_boots, 97.5))
    kap_lo = float(np.nanpercentile(kap_boots, 2.5))
    kap_hi = float(np.nanpercentile(kap_boots, 97.5))

    # Chi-square test of independence
    from scipy.stats import chi2_contingency
    from sklearn.metrics import confusion_matrix as _cm_fn
    cm = _cm_fn(y_true, y_pred, labels=labels)
    chi2_stat, p_chi2, dof, _ = chi2_contingency(cm)

    # ---- clean-switch sub-analysis (Pro↔Con only, no Tie intermediary) ----
    clean = sub[
        ((sub["agreed_before"] == "Pro") & (sub["agreed_after"] == "Con")) |
        ((sub["agreed_before"] == "Con") & (sub["agreed_after"] == "Pro"))
    ].copy()
    n_clean = len(clean)
    if n_clean >= 5:
        acc_clean = accuracy(clean["switch_direction"].tolist(), clean["claude_judgment"].tolist())
        kap_clean = cohen_kappa(clean["switch_direction"].tolist(), clean["claude_judgment"].tolist(), labels=labels)
        cm_clean = _cm_fn(clean["switch_direction"].tolist(), clean["claude_judgment"].tolist(), labels=labels)
    else:
        acc_clean = kap_clean = cm_clean = None

    result = {
        "n_total_switchers": n_total_switchers,
        "n_debates_with_switches": n_debates_with_switches,
        "n_2class_switchers": n_2class,
        "n_2class_debates": n_debates_2class,
        "n_clean_switchers": n_clean,
        "accuracy": round(acc * 100, 2),
        "acc_lo95": round(acc_lo * 100, 2),
        "acc_hi95": round(acc_hi * 100, 2),
        "kappa": round(kap, 4),
        "kappa_lo95": round(kap_lo, 4),
        "kappa_hi95": round(kap_hi, 4),
        "gwet_ac1": round(ac1, 4),
        "macro_f1": round(mf1, 4),
        "chi2": round(float(chi2_stat), 4),
        "chi2_p": round(float(p_chi2), 5),
        "chi2_dof": int(dof),
        "acc_clean": round(acc_clean * 100, 2) if acc_clean is not None else None,
        "kappa_clean": round(kap_clean, 4) if kap_clean is not None else None,
        "confusion_matrix": cm,
        "confusion_matrix_clean": cm_clean,
        "labels": labels,
    }

    # ---- confusion matrix plot ----
    if HAS_MPL and not no_plots:
        n_panels = 2 if (cm_clean is not None) else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]

        def _plot_cm(ax, mat, title, total):
            im = ax.imshow(mat, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([f"Claude: {l}" for l in labels], fontsize=10)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels([f"Switch→{l}" for l in labels], fontsize=10)
            ax.set_xlabel("Claude's judgment", fontsize=11)
            ax.set_ylabel("Switch direction", fontsize=11)
            ax.set_title(title, fontsize=11)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    pct = mat[i, j] / total * 100 if total else 0
                    color = "white" if mat[i, j] > mat.max() * 0.6 else "black"
                    ax.text(j, i, f"{mat[i,j]}\n({pct:.1f}%)", ha="center", va="center",
                            fontsize=9, color=color)

        _plot_cm(axes[0], cm, f"All directional switches\n(n={n_2class} switch events, {n_debates_2class} debates)", cm.sum())
        if cm_clean is not None:
            _plot_cm(axes[1], cm_clean, f"Clean Pro↔Con switches only\n(n={n_clean})", cm_clean.sum())

        plt.suptitle("Switch direction vs. Claude's better-listener judgment", fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "rq1_switch_confusion.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved rq1_switch_confusion.png")

    return result


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
# 3d. HEATMAP — 5×5 Spearman correlations (5 dims × 4 sub-votes + vote_margin)
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
    n_cols = len(HEATMAP_COLS)  # 5
    rhos = np.zeros((len(DIMS), n_cols))
    pvals = np.zeros((len(DIMS), n_cols))
    rhos_wtd = np.zeros((len(DIMS), n_cols))

    for i, dim in enumerate(DIMS):
        for j, col in enumerate(HEATMAP_COLS):
            x_col = f"margin_{dim}"
            y_col = "vote_margin" if col == "vote_margin" else f"subvote_margin_{col}"
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

            rows.append({"dim": dim, "subvote": col, "n": n,
                         "rho": round(rho, 4), "p": round(p, 5),
                         "rho_wtd": round(rho_w, 4), "p_wtd": round(p_w, 5)})

    # BH correction over all 25 cells (5×5; unweighted p)
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

        short_col = {
            "better_conduct": "Conduct",
            "better_spelling_and_grammar": "Grammar",
            "more_convincing_arguments": "Arguments",
            "most_reliable_sources": "Sources",
            "vote_margin": "Vote\nMargin",
        }
        short_dim = {
            "acknowledgment": "Acknowledge",
            "accuracy_of_representation": "Accuracy",
            "responsiveness": "Responsive",
            "concession_and_common_ground": "Concession",
            "respectful_engagement": "Respect",
        }

        def _draw_heatmap(rho_mat, title, filename, wtd=False):
            fig, ax = plt.subplots(figsize=(11, 6))
            vmax = max(abs(rho_mat.min()), abs(rho_mat.max()), 0.1)
            im = ax.imshow(rho_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            plt.colorbar(im, ax=ax, label="Spearman ρ" + (" (voter-weighted)" if wtd else ""))
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels([short_col[c] for c in HEATMAP_COLS], fontsize=10)
            ax.set_yticks(range(len(DIMS)))
            ax.set_yticklabels([short_dim[d] for d in DIMS], fontsize=10)
            ax.set_xlabel("Sub-vote / vote margin (Con − Pro)", fontsize=11)
            ax.set_ylabel("Listening dimension margin (Con − Pro)", fontsize=11)
            ax.set_title(title, fontsize=11)
            q_key = "q_bh_wtd" if wtd else "q_bh"
            for i in range(len(DIMS)):
                for j in range(n_cols):
                    r_entry = hm_df[(hm_df["dim"] == DIMS[i]) & (hm_df["subvote"] == HEATMAP_COLS[j])].iloc[0]
                    star = "★" if r_entry[q_key] < 0.05 else ""
                    text = f"{rho_mat[i,j]:.2f}{star}"
                    color = "white" if abs(rho_mat[i, j]) > vmax * 0.6 else "black"
                    ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved {Path(filename).name}")

        _draw_heatmap(
            rhos,
            "Spearman ρ: listening margin × sub-vote / vote margin\n(★ = BH-corrected q < 0.05, 25 tests)",
            output_dir / "rq1_heatmap.png",
        )
        _draw_heatmap(
            rhos_wtd,
            "Spearman ρ (voter-weighted): listening × sub-vote / vote margin\n(★ = BH-corrected q < 0.05, 25 tests)",
            output_dir / "rq1_heatmap_weighted.png",
            wtd=True,
        )

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
# 3f. PER-DIMENSION CORRELATIONS WITH BINARIZED GROUND TRUTH
# ===========================================================================

def analyze_dim_vs_ground_truth(df: pd.DataFrame, n_boot: int, output_dir: Path, no_plots: bool):
    """
    For each listening dimension margin, compute Spearman rho against three binarized
    ground truths (Con=1, Pro=0; Tie debates excluded).

    This directly answers "which listening dimensions individually predict who won"
    without multicollinearity, complementing the heatmap (which correlates against
    sub-vote margins) and the logistic regression (which suffers from correlated
    predictors). Bootstrap 95% CIs from n_boot debate-level resamples.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    GT_VARIANTS = [
        ("q1_ground_truth",        "Q1 ground truth"),
        ("majority_winner_procon",  "Majority winner (Pro/Con)"),
        ("majority_winner",         "Majority winner"),
    ]

    rows = []
    for dim in DIMS:
        x_col = f"margin_{dim}"
        for gt_col, gt_label in GT_VARIANTS:
            sub = df[[x_col, gt_col]].dropna()
            sub = sub[sub[gt_col].isin(["Pro", "Con"])].copy()
            sub["y_bin"] = (sub[gt_col] == "Con").astype(float)
            n = len(sub)
            if n < 5:
                continue
            rho, p = stats.spearmanr(sub[x_col].values, sub["y_bin"].values)
            ci_lo, ci_hi = bootstrap_ci(
                lambda d: stats.spearmanr(d[x_col].values, d["y_bin"].values)[0],
                sub, n_boot
            )
            rows.append({
                "dimension": dim,
                "ground_truth_variant": gt_label,
                "n": n,
                "rho": round(float(rho), 4),
                "p": round(float(p), 5),
                "ci_lo_95": round(float(ci_lo), 4),
                "ci_hi_95": round(float(ci_hi), 4),
            })

    dim_gt_df = pd.DataFrame(rows)
    dim_gt_df.to_csv(output_dir / "rq1_dim_gt_correlations.csv", index=False)
    print(f"Saved rq1_dim_gt_correlations.csv ({len(dim_gt_df)} rows)")

    if HAS_MPL and not no_plots:
        import matplotlib.pyplot as plt

        short_dim = {
            "acknowledgment": "Acknowledge",
            "accuracy_of_representation": "Accuracy",
            "responsiveness": "Responsive",
            "concession_and_common_ground": "Concession",
            "respectful_engagement": "Respect",
        }
        gt_labels_ordered = [v for _, v in GT_VARIANTS]
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        x = np.arange(len(DIMS))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        for gi, gt_label in enumerate(gt_labels_ordered):
            rhos_gt, errs_lo, errs_hi = [], [], []
            for dim in DIMS:
                r = dim_gt_df[(dim_gt_df["dimension"] == dim) &
                               (dim_gt_df["ground_truth_variant"] == gt_label)]
                if len(r) == 0:
                    rhos_gt.append(0.0)
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
                else:
                    r = r.iloc[0]
                    rhos_gt.append(r["rho"])
                    errs_lo.append(r["rho"] - r["ci_lo_95"])
                    errs_hi.append(r["ci_hi_95"] - r["rho"])
            offset = (gi - 1) * width
            ax.bar(x + offset, rhos_gt, width, label=gt_label,
                   color=colors[gi], alpha=0.8)
            ax.errorbar(x + offset, rhos_gt,
                        yerr=[errs_lo, errs_hi],
                        fmt="none", color="black", capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([short_dim[d] for d in DIMS], fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=12)
        ax.set_xlabel("Listening dimension", fontsize=12)
        ax.set_title(
            "Per-dimension Spearman ρ with binarized ground truth (Con=1, Pro=0)\n"
            "95% bootstrap CIs shown; Tie debates excluded",
            fontsize=11,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend(title="Ground truth", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / "rq1_dim_gt_barchart.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved rq1_dim_gt_barchart.png")

    return dim_gt_df


# ===========================================================================
# 3g. CROSS-VALIDATED LOGISTIC CLASSIFIER
# ===========================================================================

def analyze_cv_classifier(df: pd.DataFrame, output_dir: Path):
    """
    Cross-validated logistic regression predicting debate winner from eight features:
      - 5 listening-dimension margins
      - 3 binary indicators for Claude's overall judgment (Pro / Con / Tie)

    2-class condition: Q1 ground truth, Tie rows excluded from ground truth;
      all Claude judgment values (including Tie) retained as features.
    3-class condition: Q1 ground truth, Tie included; all rows with non-null Q1 label.

    GridSearchCV over C and penalty with StratifiedKFold(10, shuffle=True, seed=42).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, GridSearchCV
    except ImportError:
        print("scikit-learn not available; skipping CV classifier analysis")
        return None

    import warnings

    feat_cols = [f"margin_{d}" for d in DIMS]

    # Binary claude_overall indicators
    df2 = df.copy()
    for label in ["Pro", "Con", "Tie"]:
        df2[f"claude_{label.lower()}"] = (df2["claude_overall"] == label).astype(float)

    feature_cols = feat_cols + ["claude_pro", "claude_con", "claude_tie"]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
    }

    result_rows = []
    details = {}

    # ------------------------------------------------------------------
    # 2-class: Q1 ground truth, Tie rows excluded
    # ------------------------------------------------------------------
    sub2 = (
        df2[df2["q1_ground_truth"].isin(["Pro", "Con"])]
        .dropna(subset=feature_cols + ["q1_ground_truth"])
        .copy()
    )
    X2 = sub2[feature_cols].values
    y2 = (sub2["q1_ground_truth"] == "Con").astype(int).values
    n2 = len(sub2)
    counts2 = np.bincount(y2)
    majority_baseline_2 = round(100.0 * counts2.max() / n2, 2)

    lr2 = LogisticRegression(solver="saga", max_iter=5000, random_state=42)
    gs2 = GridSearchCV(lr2, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs2.fit(X2, y2)

    idx2 = gs2.best_index_
    cv_mean_2 = round(gs2.cv_results_["mean_test_score"][idx2] * 100, 2)
    cv_std_2  = round(gs2.cv_results_["std_test_score"][idx2]  * 100, 2)

    result_rows.append({
        "condition":              "2-class",
        "n":                      n2,
        "cv_accuracy_mean":       cv_mean_2,
        "cv_accuracy_std":        cv_std_2,
        "best_C":                 gs2.best_params_["C"],
        "best_penalty":           gs2.best_params_["penalty"],
        "heuristic_accuracy":     64.37,
        # always_majority_class_pct: accuracy of a classifier that always predicts
        # whichever class (Con or Pro) is most frequent in the 2-class subset
        "always_majority_class_pct": majority_baseline_2,
        "random_baseline":        50.0,
        "rescala_gpt4":           None,
        "rescala_majority_vote":  None,
    })
    details["2class"] = {
        "n": n2, "cv_mean": cv_mean_2, "cv_std": cv_std_2,
        "best_C": gs2.best_params_["C"], "best_penalty": gs2.best_params_["penalty"],
        "heuristic_accuracy": 64.37, "majority_baseline": majority_baseline_2, "random_baseline": 50.0,
    }

    # ------------------------------------------------------------------
    # 3-class: Q1 ground truth including Tie
    # ------------------------------------------------------------------
    sub3 = (
        df2[df2["q1_ground_truth"].isin(["Pro", "Con", "Tie"])]
        .dropna(subset=feature_cols + ["q1_ground_truth"])
        .copy()
    )
    X3 = sub3[feature_cols].values
    y3_labels = sub3["q1_ground_truth"].values
    n3 = len(sub3)
    # Create integer codes for bincount
    y3_encoded = pd.Categorical(y3_labels, categories=["Pro", "Con", "Tie"]).codes
    counts3 = np.bincount(y3_encoded[y3_encoded >= 0])  # filter out any -1 (missing)
    majority_baseline_3 = round(100.0 * counts3.max() / n3, 2)

    # Use the original labels for the model
    lr3 = LogisticRegression(solver="saga", max_iter=5000, random_state=42)
    gs3 = GridSearchCV(lr3, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs3.fit(X3, y3_labels)

    idx3 = gs3.best_index_
    cv_mean_3 = round(gs3.cv_results_["mean_test_score"][idx3] * 100, 2)
    cv_std_3  = round(gs3.cv_results_["std_test_score"][idx3]  * 100, 2)

    result_rows.append({
        "condition":              "3-class",
        "n":                      n3,
        "cv_accuracy_mean":       cv_mean_3,
        "cv_accuracy_std":        cv_std_3,
        "best_C":                 gs3.best_params_["C"],
        "best_penalty":           gs3.best_params_["penalty"],
        "heuristic_accuracy":     48.14,
        "always_majority_class_pct": majority_baseline_3,
        "random_baseline":        RESCALA_RANDOM,
        "rescala_gpt4":           RESCALA_GPT4,
        "rescala_majority_vote":  RESCALA_MAJORITY,
    })
    details["3class"] = {
        "n": n3, "cv_mean": cv_mean_3, "cv_std": cv_std_3,
        "best_C": gs3.best_params_["C"], "best_penalty": gs3.best_params_["penalty"],
        "heuristic_accuracy": 48.14, "majority_baseline": majority_baseline_3, "random_baseline": RESCALA_RANDOM,
    }

    clf_df = pd.DataFrame(result_rows)
    clf_df.to_csv(output_dir / "rq1_classifier.csv", index=False)
    print(f"Saved rq1_classifier.csv")

    details["df"] = clf_df
    return details


# ===========================================================================
# 4. REPORT
# ===========================================================================

def write_report(
    df, metrics_df, sw_df, sw_cond, hm_df, cont, logit_df,
    dim_gt_df, clf_result, output_dir: Path
):
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
        "The 5×5 heatmap reports uncorrected Spearman ρ with p-values; "
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

    # -----------------------------------------------------------------------
    # NEW SECTION: Per-Dimension Correlations with Binarized Ground Truth
    # -----------------------------------------------------------------------
    h(2, "Per-Dimension Correlations with Binarized Ground Truth")
    p(
        "To determine which listening dimensions individually predict who won, "
        "we compute Spearman ρ between each dimension margin (Con − Pro) and three "
        "binarized ground truths (Con = 1, Pro = 0; Tie debates excluded in all variants). "
        "This avoids the multicollinearity problem of the joint logistic regression. "
        "Bootstrap 95% CIs from 2000 debate-level resamples."
    )
    blank()

    if dim_gt_df is not None and len(dim_gt_df):
        p("| Dimension | Ground truth variant | n | ρ | p | 95% CI |")
        p("|---|---|---|---|---|---|")
        for _, r in dim_gt_df.iterrows():
            p(f"| {r['dimension']} | {r['ground_truth_variant']} | {r['n']} | "
              f"{r['rho']:.4f} | {r['p']:.5f} | "
              f"[{r['ci_lo_95']:.4f}, {r['ci_hi_95']:.4f}] |")
        blank()
    else:
        p("*(dimension–ground-truth correlation table not available)*")
        blank()

    p("![Per-dimension ρ bar chart](rq1_dim_gt_barchart.png)")
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

    # --- Conditional switchers analysis ---
    h(2, "Vote Switching: Conditional on Switching")
    p(
        "The correlation analysis above tests whether debates where Con listened better "
        "also saw *more* net switching toward Con — but most voters do not switch at all, "
        "which dilutes the signal. "
        "This section asks a sharper question: **among voters who actually switched their vote, "
        "did they tend to move toward the side Claude identified as the better listener?** "
        "The unit of analysis here is the individual switch event (voter × debate). "
        "Bootstrap CIs resample at the debate level to respect within-debate clustering "
        "(all switchers in a debate share the same Claude judgment)."
    )
    blank()
    n_sw = sw_cond["n_total_switchers"]
    n_d_sw = sw_cond["n_debates_with_switches"]
    n_2c = sw_cond["n_2class_switchers"]
    n_d_2c = sw_cond["n_2class_debates"]
    p(f"Total switch events across all {len(df)} debates: {n_sw} "
      f"(from {n_d_sw} debates that had at least one vote flip). "
      f"After excluding debates where Claude's judgment was Tie: "
      f"{n_2c} switch events across {n_d_2c} debates.")
    blank()

    h(3, "Headline: switch direction vs. Claude judgment (Tie debates excluded)")
    p("| n switches | n debates | Accuracy (%) | 95% CI | Cohen's κ | 95% CI | Gwet's AC1 | Macro F1 |")
    p("|---|---|---|---|---|---|---|---|")
    p(f"| {n_2c} | {n_d_2c} | {sw_cond['accuracy']:.2f} | "
      f"[{sw_cond['acc_lo95']:.2f}, {sw_cond['acc_hi95']:.2f}] | "
      f"{sw_cond['kappa']:.4f} | [{sw_cond['kappa_lo95']:.4f}, {sw_cond['kappa_hi95']:.4f}] | "
      f"{sw_cond['gwet_ac1']:.4f} | {sw_cond['macro_f1']:.4f} |")
    blank()

    p(f"Chi-square test of independence (switch direction × Claude judgment): "
      f"χ²({sw_cond['chi2_dof']}) = {sw_cond['chi2']:.4f}, p = {sw_cond['chi2_p']:.5f}.")
    blank()

    if sw_cond["acc_clean"] is not None:
        n_cl = sw_cond["n_clean_switchers"]
        h(3, "Robustness: clean Pro↔Con switches only")
        p(
            f"Restricting to voters who switched cleanly between Pro and Con "
            f"(excluding Pro↔Tie and Tie↔Con transitions): "
            f"n = {n_cl} switch events. "
            f"Accuracy = {sw_cond['acc_clean']:.2f}%, Cohen's κ = {sw_cond['kappa_clean']:.4f}."
        )
        blank()

    p("![Switch confusion matrix](rq1_switch_confusion.png)")
    blank()

    # --- Heatmap ---
    h(2, "Per-Dimension × Per-Sub-Vote Heatmap (5×5)")
    p(
        "The following heatmap shows Spearman ρ between each listening-dimension margin "
        "(Con − Pro mean score) and each persuasion sub-vote margin "
        "((n_Con − n_Pro) / n_votes on that sub-vote), plus the overall post-debate vote margin "
        "(the 5th column). "
        "This is an exploratory 5×5 analysis (25 simultaneous tests); "
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
        p("| Listening dim | Sub-vote / vote margin | n | ρ | p | q |")
        p("|---|---|---|---|---|---|")
        for _, r in sig_cells.iterrows():
            p(f"| {r['dim']} | {r['subvote']} | {r['n']} | {r['rho']:.4f} | {r['p']:.5f} | {r['q_bh']:.5f} |")
    else:
        p("No cells survive BH correction at q < 0.05. The heatmap is exploratory.")
    blank()

    # Full heatmap table
    p("**Full heatmap values:**")
    blank()
    p("| Listening dim | Sub-vote / vote margin | n | ρ | p | q (BH) |")
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

    # -----------------------------------------------------------------------
    # NEW SECTION: Cross-Validated Logistic Classifier
    # -----------------------------------------------------------------------
    h(2, "Cross-Validated Logistic Classifier")
    p(
        "To assess whether listening features can predict the debate winner beyond "
        "Claude's own categorical judgment, we train a logistic regression classifier "
        "using eight features: the five listening-dimension margins plus three binary indicators "
        "for Claude's overall judgment (one each for Pro, Con, Tie — set to 1 if Claude's overall "
        "judgment matches that label, 0 otherwise). "
        "Model selection uses GridSearchCV over C ∈ {0.001, 0.01, 0.1, 1, 10, 100} and "
        "penalty ∈ {L1, L2} with solver='saga' and max_iter=5000. "
        "Cross-validation uses StratifiedKFold(n_splits=10, shuffle=True, random_state=42). "
        "Accuracy is the mean CV score at the best hyperparameter combination."
    )
    blank()
    p(
        "**2-class condition:** Q1 ground truth (Con = 1, Pro = 0); Tie debates excluded from the "
        "ground truth label but Claude's Tie judgments are retained as features. "
        "**3-class condition:** Q1 ground truth with Tie included as a third class; "
        "all debates with a non-null Q1 label."
    )
    blank()

    if clf_result is not None:
        c2 = clf_result["2class"]
        c3 = clf_result["3class"]
        p(
            "Baselines: *heuristic* = Claude's single overall-judgment label used directly as a "
            "classifier (no learning); *always-majority* = accuracy of always "
            "predicting the most frequent class in the dataset; *random* = uniform random "
            "over classes. Rescala benchmarks apply to 3-class only and are on a related task "
            "(see Methods Note)."
        )
        blank()
        p("| Condition | n | CV accuracy (%) | ±std | Best C | Best penalty | "
          "Heuristic (%) | Always-majority (%) | Random (%) | "
          "Rescala GPT-4 (%) | Rescala majority (%) |")
        p("|---|---|---|---|---|---|---|---|---|---|---|")
        p(
            f"| 2-class | {c2['n']} | {c2['cv_mean']:.2f} | {c2['cv_std']:.2f} | "
            f"{c2['best_C']} | {c2['best_penalty']} | "
            f"{c2['heuristic_accuracy']:.2f} | {c2['majority_baseline']:.2f} | {c2['random_baseline']:.2f} | — | — |"
        )
        p(
            f"| 3-class | {c3['n']} | {c3['cv_mean']:.2f} | {c3['cv_std']:.2f} | "
            f"{c3['best_C']} | {c3['best_penalty']} | "
            f"{c3['heuristic_accuracy']:.2f} | {c3['majority_baseline']:.2f} | {c3['random_baseline']:.2f} | "
            f"{RESCALA_GPT4:.2f} | {RESCALA_MAJORITY:.2f} |"
        )
        blank()
        p(
            f"The 2-class CV accuracy ({c2['cv_mean']:.2f}%) compares to the "
            f"single-judgment heuristic ({c2['heuristic_accuracy']:.2f}%), the "
            f"always-majority baseline ({c2['majority_baseline']:.2f}% — always predict "
            f"the more frequent class in the 2-class subset), and to "
            f"the random baseline ({c2['random_baseline']:.2f}%), "
            f"The 3-class CV accuracy ({c3['cv_mean']:.2f}%) compares to the "
            f"single-judgment heuristic ({c3['heuristic_accuracy']:.2f}%), the "
            f"always-majority baseline ({c3['majority_baseline']:.2f}% — always predict "
            f"the more frequent class in the 3-class subset), "
            f"the random baseline ({c3['random_baseline']:.2f}%), "
            f"Rescala et al.'s GPT-4 benchmark ({RESCALA_GPT4:.2f}%), "
            f"and their majority-vote baseline ({RESCALA_MAJORITY:.2f}%). "
            "Note that the Rescala benchmarks are on a related but not identical task; "
            "comparisons are contextual."
        )
        blank()
        p("Full results saved to `rq1_classifier.csv`.")
    else:
        p("*(CV classifier results not available — scikit-learn may not be installed)*")
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
        p(f"BH-significant cells in the weighted heatmap: {len(sig_wtd)}/25.")
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
    parser.add_argument("--reuse-df", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_load = False
    if args.reuse_df:
        df_path = output_dir / "rq1_joined.csv"
        if df_path.exists():
            try:
                df = pd.read_csv(df_path)
                print(f"Successfully loaded existing dataframe from {df_path} ({len(df)} rows)")
                successful_load = True
            except Exception as e:
                print(f"Did not find valid df at {df_path}: {e}")
        else:
            print(f"Did not find file at {df_path}")

    if not successful_load:
        print("Building dataframe...")
        df = build_dataframe()
        df.to_csv(output_dir / "rq1_joined.csv", index=False)
        print(f"Saved rq1_joined.csv ({len(df)} rows)")

    print(f"\nRunning headline analysis (bootstrap n={args.bootstrap})...")
    metrics_df = analyze_winner_agreement(df, args.bootstrap, output_dir, args.no_plots)

    print("Running vote-switching analysis...")
    sw_df = analyze_vote_switching(df, args.bootstrap, output_dir, args.no_plots)

    print("Running conditional vote-switching analysis...")
    sw_cond = analyze_switchers_conditional(df, args.bootstrap, output_dir, args.no_plots)

    print("Running continuous analysis...")
    cont = analyze_continuous(df, args.bootstrap)

    print("Running heatmap analysis (5×5)...")
    hm_df = analyze_heatmap(df, args.bootstrap, output_dir, args.no_plots)

    print("Running logistic regression...")
    logit_df = analyze_logistic(df)

    print("Running per-dimension vs. ground-truth correlations...")
    dim_gt_df = analyze_dim_vs_ground_truth(df, args.bootstrap, output_dir, args.no_plots)

    print("Running cross-validated logistic classifier...")
    clf_result = analyze_cv_classifier(df, output_dir)

    print("Writing report...")
    write_report(
        df, metrics_df, sw_df, sw_cond, hm_df, cont, logit_df,
        dim_gt_df, clf_result, output_dir
    )

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
