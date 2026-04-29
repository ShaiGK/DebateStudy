# Debate Study: LLM Listening Judgments and Debate Outcomes

Undergraduate thesis code. The project asks: can a large language model identify which side of an online debate "listened" better — and does that judgment predict who won?

The corpus is the debate.org dataset curated by Rescala et al. (2024) — 833 multi-round debates with pre- and post-debate votes from readers. This repo adds a five-dimension **listening rubric**, a human pilot annotation tool, a Claude evaluation pipeline, inter-annotator agreement analyses, correlational analyses against persuasion outcomes, and a cross-validated classifier that uses listening features to predict debate winners.

The work is organized around four research questions:

- **RQ0 — Validity.** Does Claude's listening judgment agree with trained human annotators on a 20-debate pilot? (IAA: 87.5% accuracy, Cohen's κ = 0.750 on 2-class)
- **RQ1 — Winner agreement.** Does the side Claude rates as the better listener match the side that actually won the vote? (64.37% 2-class accuracy vs. Q1 ground truth, n = 595)
- **RQ2 — Vote switching.** Among voters who switched their stance, did they tend to move toward the better listener? (56.54% of 283 switch events, p = 0.041)
- **RQ3 — Classifier.** Can the five listening-dimension scores, fed to a cross-validated logistic regression, predict debate winners competitively with purpose-built persuasion models? (57.15% 3-class CV accuracy vs. Rescala et al.'s 60.50% GPT-4 baseline)

---

## The listening rubric

Each debater is scored on five dimensions. Four use a 1–5 scale; `concession_and_common_ground` uses 1–3.

| Dimension | Question |
|---|---|
| `acknowledgment` | Does the debater explicitly reference or engage with the opponent's specific arguments? |
| `accuracy_of_representation` | When referenced, are the opponent's arguments represented fairly? |
| `responsiveness` | Does the debater adapt across rounds in response to the opponent? |
| `concession_and_common_ground` | Does the debater concede valid points or identify agreement? (1–3) |
| `respectful_engagement` | Does the debater engage respectfully with the opposing perspective? |

Plus an **overall better listener** judgment: `Pro`, `Con`, or `Tie`. The full rubric with score anchors is in [`listening_evaluation_prompt_template.md`](listening_evaluation_prompt_template.md); iterative development history is in [`RUBRIC_CHANGELOG.md`](RUBRIC_CHANGELOG.md).

---

## Repository layout

```
debate_study/
├── app.py                     # Flask app for human pilot annotation
├── sample_pilot.py            # Pick a stratified pilot sample to annotate
├── try_prompt.py              # Run Claude on one or a few debates (prompt iteration)
├── run_claude_batch.py        # Run Claude on all 833 debates (sync or Batch API)
├── compare.py                 # RQ0: IAA between human and Claude annotations
├── rq_analysis.py            # RQ1–RQ3: all correlation and classifier analyses
├── data_loader.py             # Reads the Rescala et al. processed data
├── config.py                  # Centralized paths (no secrets)
├── prompt_templates/          # System + user prompts sent to Claude
│   ├── listening_system.txt   #   full v3 rubric + scoring instructions
│   └── listening_user.txt     #   debate payload + JSON output contract
├── listening_evaluation_prompt_template.md  # Human-readable rubric
├── RUBRIC_CHANGELOG.md        # v1 → v2 → v3 rubric development history
├── templates/, static/        # HTML + CSS for the Flask annotator
├── annotations.json           # Human pilot annotations (20 debates)
├── claude_listening.json      # Claude v3 annotations, all 833 debates
├── claude_listening_trial.json# Claude outputs from prompt-iteration runs
├── reports/
│   ├── iaa/                   # RQ0 outputs (compare.py)
│   └── rq/                   # RQ1–RQ3 outputs (rq_analysis.py)
└── requirements.txt
```

---

## Data dependencies

This repo ships with the annotation outputs (`annotations.json`, `claude_listening.json`, `claude_listening_trial.json`) but **not** with the underlying Rescala et al. processed data, which lives in the parent thesis directory. `config.py` expects the following layout:

```
<parent>/
├── debate_study/                            # this repo
└── data/
    ├── processing/
    │   ├── filtered_data/
    │   │   ├── debates_filtered_df.json     # ~830 valid debates + metadata
    │   │   └── votes_filtered_df.json       # voter-level pre/post votes
    │   ├── processed_data/
    │   │   ├── rounds_df.json               # debate text by round and side
    │   │   └── users_df.json
    │   └── propositions/
    │       └── propositions.json            # one proposition per debate
    └── tidy/
        ├── datasets/datasets.json           # the "Trimmed" 833-debate ID set
        └── llm_outputs/q1.json              # Q1 ground-truth winners (Rescala)
```

These files come from the Rescala et al. (2024) preprocessing pipeline. Without them, the analysis scripts (`rq_analysis.py`, `compare.py`) will raise `FileNotFoundError` when they try to join listening scores with vote outcomes.

If you only want to inspect the annotation outputs, `annotations.json` and `claude_listening.json` are self-contained and do not require the data files.

---

## Setup

Requires Python 3.10+.

```bash
cd debate_study
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

Verify the data link is in place:

```bash
python -c "from data_loader import get_valid_debate_ids; print(len(get_valid_debate_ids()))"
# expected: 833
```

Libraries: flask, pandas, numpy, scipy, scikit-learn, matplotlib, statsmodels, anthropic, python-dotenv.

---

## Typical workflow

### 1. Pick a pilot sample and annotate it (human side of RQ0)

```bash
python sample_pilot.py --n 20 --min-votes 5 --seed 7 --write pilot_ids.json
python app.py    # open http://127.0.0.1:5000
```

`sample_pilot.py` selects debates stratified by Q1 ground-truth class (Pro / Con / Tie) to ensure the pilot covers the full range of outcomes, excluding any already in `annotations.json`. The Flask app walks through each debate and records structured per-dimension scores plus an overall better-listener judgment.

### 2. Iterate on the Claude prompt

Edit [`prompt_templates/listening_system.txt`](prompt_templates/listening_system.txt) and [`prompt_templates/listening_user.txt`](prompt_templates/listening_user.txt), then:

```bash
python try_prompt.py --debate-ids 358,412,900 --save-results
python try_prompt.py --debate-ids 358,412,900 --max-workers 3
```

Results go to `claude_listening_trial.json` when `--save-results` is set.

### 3. Run Claude on the full dataset

```bash
# Synchronous:
python run_claude_batch.py

# Message Batches API (50% discount, async):
python run_claude_batch.py --use-batch --batch-poll-secs 30
python run_claude_batch.py --model claude-sonnet-4-6 --limit 10   # smoke test
```

Appends to `claude_listening.json`. Each record includes rubric scores, overall judgment, model, token usage, and request/response metadata. Progress checkpoints go to `reports/run_claude_batch_*.json`.

### 4. RQ0 — human vs. Claude agreement

```bash
python compare.py --use-full --bootstrap 2000
```

Writes to `reports/iaa/`:

| File | Contents |
|---|---|
| `iaa_report.md` | Headline accuracy, Cohen's κ, Gwet's AC1, Krippendorff's α, weighted κ per dimension, bootstrap CIs |
| `iaa_per_dimension.csv` | Per-cell metrics (10 cells: 5 dims × Pro/Con) |
| `iaa_disagreements.md` | Every disagreement shown side by side |
| `iaa_heatmap.png` | Per-dimension weighted-κ heatmap |

Use `--use-trials` to compare against `claude_listening_trial.json` while iterating on the prompt.

### 5. RQ1–RQ3 — listening ↔ persuasion outcomes

```bash
python rq_analysis.py
```

Writes to `reports/rq/`:

| File | Contents |
|---|---|
| `rq_report.md` | Full markdown write-up with all tables and figures |
| `rq_joined.csv` | Master feature table: one row per debate, all listening scores + outcome metrics |
| `rq_overall_metrics.csv` | RQ1 winner-agreement: accuracy / κ / AC1 / macro-F1 for 3 ground truths × {2-class, 3-class} × {unweighted, voter-weighted} |
| `rq_dim_gt_correlations.csv` | RQ1 per-dimension Spearman ρ against 3 binarized ground truths |
| `rq_switching.csv` | RQ2 correlational: composite and per-dimension ρ vs. net_switch_toward_con |
| `rq_switchers_conditional.csv` | RQ2 conditional: voter-level switch events with direction + Claude judgment |
| `rq_heatmap_cells.csv` | 5×5 Spearman ρ matrix with BH-corrected q-values (unweighted + voter-weighted) |
| `rq_classifier.csv` | RQ3: cross-validated logistic classifier summary (accuracy, best C, best penalty) |
| `rq_winner_confusion.png` | RQ1 confusion matrices (3 ground truths × 2 conditions) |
| `rq_winner_confusion_weighted.png` | Voter-weighted version |
| `rq_dim_gt_barchart.png` | RQ1 grouped bar chart: per-dimension ρ vs. 3 ground truths |
| `rq_switch_scatter.png` | RQ2 scatter: composite listening margin vs. net switch toward Con |
| `rq_switch_confusion.png` | RQ2 conditional: confusion matrix of switch direction vs. Claude judgment |
| `rq_heatmap.png` | 5×5 Spearman ρ heatmap (unweighted) |
| `rq_heatmap_weighted.png` | Voter-weighted version |

---

## Conventions worth knowing before reading the code

**Sign convention.** All margins are Con-positive: `margin = con_score − pro_score` for listening dimensions, and `(n_Con − n_Pro) / n_votes` for vote margins. Positive = Con.

**Vote switching direction.** The post-debate vote options form an ordered scale: Pro < Tie < Con. A voter switches "toward Con" on any upward movement (Pro→Con, Pro→Tie, Tie→Con); the mirror set counts as switching toward Pro. `net_switch_toward_con = (n_toward_con − n_toward_pro) / n_votes`.

**Two majority-winner definitions.** `majority_winner` takes the three-way plurality among Pro / Con / Tie post-debate votes. `majority_winner_procon` ignores Tie votes and awards the debate to whichever of Pro/Con has more (Tie only when n_Pro = n_Con). The Pro/Con-only variant is the more interpretable default.

**Unit of analysis.** The debate, unweighted, is the headline. Voter-weighted variants (each debate replicated by its `n_votes`) are reported as robustness throughout. Bootstrap CIs always resample at the debate level, then expand by vote counts for weighted stats.

**Heatmap is 5×5 and exploratory.** Five listening dimension margins × five persuasion outcomes (four sub-vote margins + overall vote margin). Benjamini–Hochberg FDR correction over the 25 cells; starred cells survive q < 0.05. Do not over-read individual cells.

**Classifier feature set.** The RQ3 cross-validated logistic regression uses 8 features: the 5 listening-dimension margins plus 3 binary indicator variables derived from Claude's overall better-listener judgment (is_pro, is_con, is_tie). StratifiedKFold with 10 folds; hyperparameter search over C and penalty (L1/L2).

---

## Citing the underlying data

Rescala, P. et al. (2024). *Can language models recognize convincing arguments?* The 833-debate Trimmed corpus and Q1 ground truth are theirs. Their Table 2 benchmarks (33.33% random baseline, 60.69% majority-vote baseline, 60.50% GPT-4) are used as comparison points in the RQ1 and RQ3 analyses.

---

## Troubleshooting

- **`FileNotFoundError` on a `data/processing/...` path** — the Rescala et al. data tree is not in place under `<parent>/data/`. See "Data dependencies" above. The annotation JSON files in this repo work without it, but the analysis scripts require the vote and debate metadata.
- **`ANTHROPIC_API_KEY` missing** — copy `.env.example` to `.env` and fill it in; scripts load it via `python-dotenv`.
- **`get_valid_debate_ids()` returns fewer than 833** — `datasets.json` is missing or the path in `config.py` is wrong; the `"Trimmed"` key defines the valid debate set.
- **Batch run appears stuck** — `--use-batch` polls the Message Batches API on an interval; current status is written to `reports/run_claude_batch_*.json`. Check that file rather than killing the process.
