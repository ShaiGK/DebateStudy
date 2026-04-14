"""
Flask app: list debates, view one debate, submit structured listening annotations.
"""
import json
from datetime import datetime

from flask import Flask, redirect, render_template, request, url_for

from config import ANNOTATIONS_PATH
from data_loader import get_valid_debate_ids, get_debate

app = Flask(__name__)

LISTENING_DIMENSIONS = [
    {
        "key": "acknowledgment",
        "label": "Acknowledgment",
        "question": "Does the debater explicitly reference or engage with the opponent's specific arguments?",
        "scores": [1, 2, 3, 4, 5],
    },
    {
        "key": "accuracy_of_representation",
        "label": "Accuracy of Representation",
        "question": "When the debater references the opponent, are those arguments represented fairly and accurately?",
        "scores": [1, 2, 3, 4, 5],
    },
    {
        "key": "responsiveness",
        "label": "Responsiveness / Adaptation",
        "question": "Does the debater adapt arguments across rounds in response to the opponent?",
        "scores": [1, 2, 3, 4, 5],
    },
    {
        "key": "concession_and_common_ground",
        "label": "Concession and Common Ground",
        "question": "Does the debater acknowledge valid points, concede where appropriate, or identify agreement?",
        "scores": [1, 2, 3],
    },
    {
        "key": "respectful_engagement",
        "label": "Respectful Engagement",
        "question": "Does the debater engage respectfully with the opponent's perspective?",
        "scores": [1, 2, 3, 4, 5],
    },
]
VALID_JUDGMENTS = ("Pro", "Con", "Tie")
# Per-dimension valid score sets derived from LISTENING_DIMENSIONS.
_DIM_VALID_SCORES = {d["key"]: set(d["scores"]) for d in LISTENING_DIMENSIONS}


def _load_annotations() -> dict:
    if not ANNOTATIONS_PATH.exists():
        return {}
    with open(ANNOTATIONS_PATH, "r") as f:
        return json.load(f)


def _save_annotations(data: dict) -> None:
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _empty_listening_payload() -> dict:
    payload = {}
    for side in ("pro", "con"):
        payload[side] = {
            d["key"]: {"score": "", "justification": ""} for d in LISTENING_DIMENSIONS
        }
    payload["overall_better_listener"] = {"judgment": "", "justification": ""}
    payload["notes"] = ""
    return payload


def _normalize_listening_payload(raw: dict) -> dict:
    payload = _empty_listening_payload()

    if not isinstance(raw, dict):
        return payload

    for side in ("pro", "con"):
        if not isinstance(raw.get(side), dict):
            continue
        for d in LISTENING_DIMENSIONS:
            dim = raw[side].get(d["key"], {})
            if not isinstance(dim, dict):
                continue
            score = dim.get("score", "")
            valid = _DIM_VALID_SCORES[d["key"]]
            if isinstance(score, int) and score in valid:
                payload[side][d["key"]]["score"] = score
            elif isinstance(score, str) and score.isdigit() and int(score) in valid:
                payload[side][d["key"]]["score"] = int(score)
            payload[side][d["key"]]["justification"] = str(dim.get("justification", "")).strip()

    overall = raw.get("overall_better_listener", {})
    if isinstance(overall, dict):
        judgment = overall.get("judgment", "")
        if judgment in VALID_JUDGMENTS:
            payload["overall_better_listener"]["judgment"] = judgment
        payload["overall_better_listener"]["justification"] = str(
            overall.get("justification", "")
        ).strip()

    # Backward compatibility for old schema that only stored "judgment" and "notes".
    legacy_judgment = raw.get("judgment", "")
    if not payload["overall_better_listener"]["judgment"] and legacy_judgment in VALID_JUDGMENTS:
        payload["overall_better_listener"]["judgment"] = legacy_judgment
    payload["notes"] = str(raw.get("notes", "")).strip()

    return payload


@app.route("/")
def index():
    ids = get_valid_debate_ids()
    if not ids:
        return "<p>No valid debates found. Check data paths.</p>", 500
    # Redirect to first debate by default; could instead render a list
    return redirect(url_for("debate", debate_id=ids[0]))


@app.route("/list")
def list_debates():
    ids = get_valid_debate_ids()
    debates = []
    ann = _load_annotations()

    for did in ids:
        listening = _normalize_listening_payload(ann.get(str(did), {}).get("listening", {}))
        is_annotated = listening.get("overall_better_listener", {}).get("judgment", {}) != ""
        d = get_debate(did)
        if d:
            debates.append({"debate_id": did, "title": d.get("title", "") or f"Debate {did}", "is_annotated": is_annotated})
    return render_template("index.html", debates=debates, annotations=ann)


@app.route("/debate/<int:debate_id>")
def debate(debate_id: int):
    debate_data = get_debate(debate_id)
    if not debate_data:
        return f"Debate {debate_id} not found.", 404

    ids = get_valid_debate_ids()
    idx = next((i for i, d in enumerate(ids) if d == debate_id), 0)
    prev_id = ids[idx - 1] if idx > 0 else None
    next_id = ids[idx + 1] if idx < len(ids) - 1 else None

    ann = _load_annotations()
    listening = _normalize_listening_payload(ann.get(str(debate_id), {}).get("listening", {}))
    return render_template(
        "debate.html",
        debate=debate_data,
        prev_id=prev_id,
        next_id=next_id,
        total=len(ids),
        current_index=idx + 1,
        listening=listening,
        listening_dimensions=LISTENING_DIMENSIONS,
        valid_judgments=VALID_JUDGMENTS,
    )


@app.route("/annotate", methods=["POST"])
def annotate():
    debate_id = request.form.get("debate_id", type=int)
    if debate_id is None:
        return "Missing debate_id.", 400

    task = request.form.get("task", "listening").strip() or "listening"
    overall_judgment = request.form.get("overall_judgment", "").strip()
    overall_justification = request.form.get("overall_justification", "").strip()
    notes = request.form.get("notes", "").strip()

    if overall_judgment not in VALID_JUDGMENTS:
        return "Invalid overall judgment.", 400

    listening_payload = {
        "pro": {},
        "con": {},
        "overall_better_listener": {
            "judgment": overall_judgment,
            "justification": overall_justification,
        },
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    }

    for side in ("pro", "con"):
        for d in LISTENING_DIMENSIONS:
            score_key = f"{side}_{d['key']}_score"
            justification_key = f"{side}_{d['key']}_justification"
            score_raw = request.form.get(score_key, "").strip()
            if not score_raw.isdigit() or int(score_raw) not in _DIM_VALID_SCORES[d["key"]]:
                return f"Invalid score for {side} {d['key']}.", 400
            listening_payload[side][d["key"]] = {
                "score": int(score_raw),
                "justification": request.form.get(justification_key, "").strip(),
            }

    data = _load_annotations()
    key = str(debate_id)
    if key not in data:
        data[key] = {}
    data[key][task] = listening_payload
    _save_annotations(data)

    next_id = request.form.get("next_id", type=int)
    if next_id is not None:
        return redirect(url_for("debate", debate_id=next_id))
    return redirect(url_for("debate", debate_id=debate_id))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
