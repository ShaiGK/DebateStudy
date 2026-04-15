"""
Load debate data and outcomes. No dependency on debate_gpt_x_main.
Reads from Thesis/data/processing/ (config.DATA_DIR).
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from config import (
    DEBATES_FILTERED_PATH,
    PROPOSITIONS_PATH,
    ROUNDS_PATH,
    VOTES_FILTERED_PATH,
    Q1_OUTPUT_PATH,
    TRIMMED_DATASET_PATH,
)


def _load_debates_df() -> pd.DataFrame:
    """Debates filtered by original research (rounds, votes, etc.)."""
    return pd.read_json(DEBATES_FILTERED_PATH)


def _load_propositions_df() -> pd.DataFrame:
    """Propositions; we exclude drop and skip."""
    df = pd.read_json(PROPOSITIONS_PATH)
    return df[(df["proposition"] != "drop") & (df["proposition"] != "skip")]


def _load_rounds_df() -> pd.DataFrame:
    """All rounds; has debate_id, round, side, text, order."""
    return pd.read_json(ROUNDS_PATH)


def _load_votes_df() -> pd.DataFrame:
    """Filtered votes: debate_id, voter_id, agreed_before, agreed_after, flipped, etc."""
    return pd.read_json(VOTES_FILTERED_PATH)


def _load_ground_truth_dict() -> dict[int, str]:
    """Ground truth for RQ1. Returns dict mapping debate_id -> ground_truth."""
    df = pd.read_json(Q1_OUTPUT_PATH)
    return {int(row["debate_id"]): row["ground_truth"] for _, row in df.iterrows()}


def get_valid_debate_ids() -> list[int]:
    """Debate IDs in the Rescala et al. Trimmed set."""
    trimmed_debates = json.load(open(TRIMMED_DATASET_PATH))["Trimmed"]
    return sorted(set(trimmed_debates))


# Cache for rounds and debates so we don't reload every time
_rounds_by_debate: dict[int, list[dict[str, Any]]] | None = None
_debates_df: pd.DataFrame | None = None
_props_df: pd.DataFrame | None = None
_ground_truth_dict: dict[int, str] | None = None


def _get_rounds_index() -> dict[int, list[dict[str, Any]]]:
    global _rounds_by_debate
    if _rounds_by_debate is None:
        df = _load_rounds_df()
        out: dict[int, list[dict[str, Any]]] = {}
        for debate_id, grp in df.groupby("debate_id", sort=False):
            rows = grp.sort_values("order")
            out[int(debate_id)] = [
                {"round": int(r["round"]), "side": r["side"], "text": r["text"]}
                for _, r in rows.iterrows()
            ]
        _rounds_by_debate = out
    return _rounds_by_debate


def _get_debates_df() -> pd.DataFrame:
    global _debates_df
    if _debates_df is None:
        _debates_df = _load_debates_df()
    return _debates_df


def _get_props_df() -> pd.DataFrame:
    global _props_df
    if _props_df is None:
        _props_df = _load_propositions_df()
    return _props_df


def _get_ground_truth_dict() -> dict[int, str]:
    global _ground_truth_dict
    if _ground_truth_dict is None:
        _ground_truth_dict = _load_ground_truth_dict()
    return _ground_truth_dict


def get_debate(debate_id: int) -> dict[str, Any] | None:
    """
    Full debate for display: metadata, proposition, rounds (sorted by order).
    Returns None if debate_id not in valid set.
    """
    valid = set(get_valid_debate_ids())
    if debate_id not in valid:
        return None

    debates_df = _get_debates_df()
    props_df = _get_props_df()
    rounds_index = _get_rounds_index()

    row = debates_df[debates_df["debate_id"] == debate_id]
    if row.empty:
        return None
    row = row.iloc[0]

    prop_row = props_df[props_df["debate_id"] == debate_id]
    proposition = ""
    if not prop_row.empty:
        proposition = str(prop_row.iloc[0]["proposition"])

    rounds = rounds_index.get(debate_id, [])

    return {
        "debate_id": int(debate_id),
        "title": row.get("title", ""),
        "category": row.get("category", ""),
        "pro_user_id": row.get("pro_user_id", ""),
        "con_user_id": row.get("con_user_id", ""),
        "proposition": proposition,
        "rounds": rounds,
    }


def format_debate_text_for_prompt(debate: dict[str, Any]) -> str:
    """Format debate rounds as JSON-like string for Claude (Round 0: {Pro: ..., Con: ...})."""
    rounds_dict: dict[str, dict[str, str]] = {}
    for r in debate["rounds"]:
        key = f"Round {r['round']}"
        if key not in rounds_dict:
            rounds_dict[key] = {}
        rounds_dict[key][r["side"]] = r["text"]
    return json.dumps(rounds_dict)


def get_debate_outcomes(debate_id: int) -> dict[str, Any] | None:
    """
    Aggregate vote outcomes for a debate (for RQ1/RQ2).
    Keys: majority_winner (Pro/Con/Tie), num_votes, num_flipped,
    pct_switched_to_pro, pct_switched_to_con, num_pro_after, num_con_after, num_tie_after.
    """
    valid = set(get_valid_debate_ids())
    if debate_id not in valid:
        return None

    votes_df = _load_votes_df()
    v = votes_df[votes_df["debate_id"] == debate_id]
    if v.empty:
        return {
            "majority_winner": "Tie",
            "majority_winner_procon": "Tie",
            "num_votes": 0,
            "num_flipped": 0,
            "pct_switched_to_pro": 0.0,
            "pct_switched_to_con": 0.0,
            "net_switch_toward_con": 0.0,
            "num_pro_after": 0,
            "num_con_after": 0,
            "num_tie_after": 0,
        }

    n = len(v)
    num_flipped = int(v["flipped"].sum()) if "flipped" in v.columns else 0
    agreed_after = v["agreed_after"] if "agreed_after" in v.columns else pd.Series([])
    num_pro_after = int((agreed_after == "Pro").sum())
    num_con_after = int((agreed_after == "Con").sum())
    num_tie_after = int((agreed_after == "Tie").sum())

    if num_pro_after > num_con_after and num_pro_after > num_tie_after:
        majority_winner = "Pro"
    elif num_con_after > num_pro_after and num_con_after > num_tie_after:
        majority_winner = "Con"
    else:
        majority_winner = "Tie"

    # Alternative: ignore Tie votes; winner is whoever has more Pro vs. Con post-votes.
    # 1 Pro + 1 Tie + 0 Con → Pro (Tie votes are neutral, not a competing outcome).
    if num_pro_after > num_con_after:
        majority_winner_procon = "Pro"
    elif num_con_after > num_pro_after:
        majority_winner_procon = "Con"
    else:
        majority_winner_procon = "Tie"

    # Switch direction: Pro→Con, Pro→Tie, Tie→Con all count as "toward Con"; mirror for Pro.
    if "agreed_before" in v.columns and "agreed_after" in v.columns:
        ab = v["agreed_before"]
        aa = v["agreed_after"]
        toward_con = (
                ((ab == "Pro") & (aa == "Con")) |
                ((ab == "Pro") & (aa == "Tie")) |
                ((ab == "Tie") & (aa == "Con"))
        )
        toward_pro = (
                ((ab == "Con") & (aa == "Pro")) |
                ((ab == "Con") & (aa == "Tie")) |
                ((ab == "Tie") & (aa == "Pro"))
        )
        n_toward_con = int(toward_con.sum())
        n_toward_pro = int(toward_pro.sum())
        pct_con = (n_toward_con / n * 100) if n else 0.0
        pct_pro = (n_toward_pro / n * 100) if n else 0.0
        net_switch_toward_con = (n_toward_con - n_toward_pro) / n if n else 0.0
    else:
        n_toward_con = n_toward_pro = 0
        pct_con = pct_pro = 0.0
        net_switch_toward_con = 0.0

    return {
        "majority_winner": majority_winner,
        "majority_winner_procon": majority_winner_procon,
        "num_votes": n,
        "num_flipped": num_flipped,
        "pct_switched_to_pro": round(pct_pro, 2),
        "pct_switched_to_con": round(pct_con, 2),
        "net_switch_toward_con": round(net_switch_toward_con, 6),
        "num_pro_after": num_pro_after,
        "num_con_after": num_con_after,
        "num_tie_after": num_tie_after,
    }


def get_votes_for_debate(debate_id: int) -> list[dict[str, Any]]:
    """Voter-level votes for RQ2 (side-switching)."""
    valid = set(get_valid_debate_ids())
    if debate_id not in valid:
        return []

    votes_df = _load_votes_df()
    v = votes_df[votes_df["debate_id"] == debate_id]
    cols = [
        "voter_id", "agreed_before", "agreed_after", "flipped",
        "better_conduct", "better_spelling_and_grammar",
        "more_convincing_arguments", "most_reliable_sources",
    ]
    available = [c for c in cols if c in v.columns]
    return v[available].to_dict("records")
