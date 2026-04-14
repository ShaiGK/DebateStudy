"""
sample_pilot.py — Suggest debates to hand-annotate for the human pilot.

Prioritizes debates with enough votes (default ≥ 5) and balances across
the Rescala et al. Q1 ground-truth class (Pro / Con / Tie) so the pilot
isn't dominated by one outcome. Excludes debates already in annotations.json.
Within each ground-truth class, prefers higher vote counts (more reliable
ground truth) and then samples randomly from the top half for diversity.

Usage:
    python sample_pilot.py                         # suggest 25 new debates
    python sample_pilot.py --n 30
    python sample_pilot.py --min-votes 5
    python sample_pilot.py --seed 7
    python sample_pilot.py --write pilot_ids.json  # also save IDs to a file
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from config import ANNOTATIONS_PATH
from data_loader import (
    get_valid_debate_ids,
    _get_ground_truth_dict,
    _load_votes_df,
)


def _already_annotated() -> set[int]:
    if not ANNOTATIONS_PATH.exists():
        return set()
    with open(ANNOTATIONS_PATH) as f:
        raw = json.load(f)
    return {int(k) for k in raw.keys()}


def _vote_counts_all() -> dict[int, dict]:
    """Load votes once and aggregate per debate, much faster than calling
    get_debate_outcomes() in a loop."""
    df = _load_votes_df()
    out: dict[int, dict] = {}
    if "debate_id" not in df.columns:
        return out
    for did, grp in df.groupby("debate_id"):
        n = len(grp)
        num_flipped = int(grp["flipped"].sum()) if "flipped" in grp.columns else 0
        out[int(did)] = {"num_votes": n, "num_flipped": num_flipped}
    return out


def suggest(n: int, min_votes: int, seed: int) -> list[dict]:
    valid_ids = get_valid_debate_ids()
    gt = _get_ground_truth_dict()
    excluded = _already_annotated()
    vote_counts = _vote_counts_all()

    candidates: list[dict] = []
    for did in valid_ids:
        if did in excluded:
            continue
        vc = vote_counts.get(did)
        if not vc or vc["num_votes"] < min_votes:
            continue
        candidates.append({
            "debate_id": did,
            "num_votes": vc["num_votes"],
            "num_flipped": vc["num_flipped"],
            "ground_truth": gt.get(did, "Unknown"),
        })

    if not candidates:
        print(f"No candidates found with num_votes >= {min_votes}.")
        return []

    # Stratify by ground truth; within each bucket, sort by votes desc
    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        buckets[c["ground_truth"]].append(c)
    for lst in buckets.values():
        lst.sort(key=lambda x: (-x["num_votes"], -x["num_flipped"]))

    classes = sorted(buckets.keys())
    base = n // len(classes)
    remainder = n - base * len(classes)
    quota = {c: base for c in classes}
    # Distribute remainder to the largest buckets first
    for c in sorted(classes, key=lambda k: -len(buckets[k]))[:remainder]:
        quota[c] += 1

    # If a bucket is smaller than its quota, redistribute overflow to buckets with room
    overflow = 0
    for c in classes:
        if quota[c] > len(buckets[c]):
            overflow += quota[c] - len(buckets[c])
            quota[c] = len(buckets[c])
    for c in sorted(classes, key=lambda k: -len(buckets[k])):
        if overflow <= 0:
            break
        room = len(buckets[c]) - quota[c]
        take = min(room, overflow)
        quota[c] += take
        overflow -= take

    rng = random.Random(seed)
    picked: list[dict] = []
    for c in classes:
        bucket = buckets[c]
        q = quota[c]
        if q == 0:
            continue
        # Sample from the top half by vote count (vote-weighted preference),
        # but shuffle so we don't always take the same deterministic top slice.
        top_pool_size = max(q, len(bucket) // 2)
        top_pool = bucket[:top_pool_size]
        rng.shuffle(top_pool)
        picked.extend(top_pool[:q])

    # Nice display order
    picked.sort(key=lambda x: (x["ground_truth"], -x["num_votes"]))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="Suggest debates for the human pilot")
    parser.add_argument("--n", type=int, default=25, help="Number of debates to suggest (default 25)")
    parser.add_argument("--min-votes", type=int, default=5, help="Minimum num_votes (default 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--write", type=str, default=None, help="Save suggested ids to JSON")
    args = parser.parse_args()

    picks = suggest(n=args.n, min_votes=args.min_votes, seed=args.seed)
    if not picks:
        return 1

    print()
    print(f"Suggested {len(picks)} debates (min_votes >= {args.min_votes}, "
          f"excluding {len(_already_annotated())} already annotated):")
    print()
    print(f"{'debate_id':>10}  {'votes':>6}  {'flipped':>8}  {'ground_truth':>12}")
    print("-" * 45)
    for p in picks:
        print(f"{p['debate_id']:>10}  {p['num_votes']:>6}  {p['num_flipped']:>8}  {p['ground_truth']:>12}")

    counts = Counter(p["ground_truth"] for p in picks)
    votes_sorted = sorted(p["num_votes"] for p in picks)
    median_votes = votes_sorted[len(votes_sorted) // 2]
    print()
    print(f"By ground-truth class: {dict(counts)}")
    print(f"Votes — min: {votes_sorted[0]}, median: {median_votes}, max: {votes_sorted[-1]}")

    if args.write:
        path = Path(args.write)
        path.write_text(json.dumps([p["debate_id"] for p in picks], indent=2))
        print(f"\nIDs saved to {path}")

    print()
    print("Next step: open each debate in the Flask app and annotate it:")
    print("  python app.py  →  http://127.0.0.1:5000/debate/<id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())