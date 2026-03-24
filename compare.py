"""
Compare human listening annotations to Claude listening output (RQ0 validation).
Usage: python compare.py [--report report.txt]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from config import ANNOTATIONS_PATH, CLAUDE_LISTENING_PATH, CLAUDE_LISTENING_TRIAL_PATH

LISTENING_DIMENSIONS = (
    "acknowledgment",
    "accuracy_of_representation",
    "responsiveness",
    "concession_and_common_ground",
    "respectful_engagement",
)
SIDES = ("pro", "con")


def _extract_overall_judgment(record: dict) -> str | None:
    if not isinstance(record, dict):
        return None

    overall = record.get("overall_better_listener", {})
    if isinstance(overall, dict):
        judgment = overall.get("judgment")
        if judgment in {"Pro", "Con", "Tie"}:
            return judgment

    judgment = record.get("overall_judgment")
    if judgment in {"Pro", "Con", "Tie"}:
        return judgment

    judgment = record.get("judgment")
    if judgment in {"Pro", "Con", "Tie"}:
        return judgment

    return None


def _extract_scores(record: dict) -> dict:
    scores = {}
    if not isinstance(record, dict):
        return scores

    for side in SIDES:
        side_obj = record.get(side, {})
        if not isinstance(side_obj, dict):
            continue
        for dim in LISTENING_DIMENSIONS:
            dim_obj = side_obj.get(dim, {})
            if not isinstance(dim_obj, dict):
                continue
            score = dim_obj.get("score")
            if isinstance(score, int) and 1 <= score <= 5:
                scores[(side, dim)] = score
            elif isinstance(score, str) and score.isdigit() and 1 <= int(score) <= 5:
                scores[(side, dim)] = int(score)
    return scores


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _normalize_listening_entries(listening) -> list[dict]:
    if isinstance(listening, list):
        return [x for x in listening if isinstance(x, dict)]
    if isinstance(listening, dict):
        return [listening]
    return []


def main():
    parser = argparse.ArgumentParser(description="Compare human vs Claude listening judgments")
    parser.add_argument("--report", type=str, default=None, help="Save text report to this file")
    parser.add_argument("--use-trials", action="store_true", help="Compare against trial outputs")
    args = parser.parse_args()

    if not ANNOTATIONS_PATH.exists():
        print(f"No annotations file: {ANNOTATIONS_PATH}")
        return 1
    claude_path = CLAUDE_LISTENING_TRIAL_PATH if args.use_trials else CLAUDE_LISTENING_PATH
    if not claude_path.exists():
        print(f"No Claude output file: {claude_path}")
        return 1

    with open(ANNOTATIONS_PATH) as f:
        ann = json.load(f)

    with open(claude_path) as f:
        claude_list = json.load(f)

    # Build human judgments: debate_id -> list[judgment] (task = listening)
    human = defaultdict(list)
    human_scores = defaultdict(list)
    for did_str, tasks in ann.items():
        listening = tasks.get("listening", {}) if isinstance(tasks, dict) else {}
        for entry in _normalize_listening_entries(listening):
            judgment = _extract_overall_judgment(entry)
            if judgment:
                human[int(did_str)].append(judgment)
            scores = _extract_scores(entry)
            if scores:
                human_scores[int(did_str)].append(scores)

    # Build Claude judgments/scores: debate_id -> list[judgment]/list[scores]
    claude = defaultdict(list)
    claude_scores = defaultdict(list)
    for r in claude_list:
        if not isinstance(r, dict) or "debate_id" not in r:
            continue
        did = int(r["debate_id"])
        judgment = _extract_overall_judgment(r)
        if not judgment and isinstance(r.get("evaluation"), dict):
            judgment = _extract_overall_judgment(r["evaluation"])
        if judgment:
            claude[did].append(judgment)

        scores = _extract_scores(r)
        if not scores and isinstance(r.get("evaluation"), dict):
            scores = _extract_scores(r["evaluation"])
        if scores:
            claude_scores[did].append(scores)

    overlap = set(human.keys()) & set(claude.keys())
    if not overlap:
        print("No overlapping debate IDs between annotations and Claude output.")
        return 0

    pair_count = 0
    matches = 0
    for did in overlap:
        for h in human[did]:
            for c in claude[did]:
                pair_count += 1
                if h == c:
                    matches += 1

    accuracy = (matches / pair_count * 100) if pair_count else 0.0
    lines = [
        f"Human annotations (listening): {sum(len(v) for v in human.values())} judgments across {len(human)} debates",
        f"Claude output (listening): {sum(len(v) for v in claude.values())} judgments across {len(claude)} debates",
        f"Overlap: {len(overlap)} debates",
        f"Overall judgment agreement: {matches} / {pair_count}",
        f"Overall judgment accuracy: {accuracy:.1f}%",
    ]

    # Confusion: human vs Claude
    confusion = defaultdict(lambda: defaultdict(int))
    for did in overlap:
        for h in human[did]:
            for c in claude[did]:
                confusion[h][c] += 1

    # Dimension MAE when both sides have numeric scores
    dim_errors = defaultdict(list)
    score_overlap = set(human_scores.keys()) & set(claude_scores.keys())
    for did in score_overlap:
        for hs in human_scores.get(did, []):
            for cs in claude_scores.get(did, []):
                for side in SIDES:
                    for dim in LISTENING_DIMENSIONS:
                        key = (side, dim)
                        if key in hs and key in cs:
                            dim_errors[key].append(abs(hs[key] - cs[key]))

    lines.append("")
    lines.append("Confusion (rows=human, cols=Claude):")
    for h in ("Pro", "Con", "Tie"):
        row = confusion[h]
        lines.append(f"  Human {h}: Pro={row['Pro']} Con={row['Con']} Tie={row['Tie']}")

    lines.append("")
    lines.append("Dimension mean absolute error (MAE; lower is better):")
    mae_values = []
    for side in SIDES:
        for dim in LISTENING_DIMENSIONS:
            key = (side, dim)
            mae = _safe_mean(dim_errors[key])
            label = f"{side}.{dim}"
            if mae is None:
                lines.append(f"  {label}: n=0, MAE=N/A")
            else:
                mae_values.append(mae)
                lines.append(f"  {label}: n={len(dim_errors[key])}, MAE={mae:.3f}")
    overall_mae = _safe_mean(mae_values)
    if overall_mae is None:
        lines.append("  Overall score MAE: N/A (no overlapping numeric scores)")
    else:
        lines.append(f"  Overall score MAE: {overall_mae:.3f}")

    report = "\n".join(lines)
    print(report)

    if args.report:
        path = Path(args.report)
        path.write_text(report + "\n")
        print(f"\nReport saved to {path}")

    return 0


if __name__ == "__main__":
    exit(main())
