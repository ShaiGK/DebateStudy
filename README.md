# Debate study (RQ0: listening)

Standalone project for RQ0 (listening scoring) with preparation for RQ1 and RQ2. Uses the same data and ~830 valid debates as the original Rescala et al. work; no dependency on their codebase.

## Setup

```bash
cd debate_study
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_key
```

Data is read from `../data/processing/` (Thesis data directory). Ensure these exist:

- `../data/processing/filtered_data/debates_filtered_df.json`
- `../data/processing/processed_data/rounds_df.json`
- `../data/processing/propositions/propositions.json`
- `../data/processing/filtered_data/votes_filtered_df.json`

## Run the web viewer and annotate

```bash
python app.py
```

Open http://127.0.0.1:5000 . You’ll see debates one at a time. In the annotation panel, score both debaters (1-5) on five listening dimensions:

- acknowledgment
- accuracy_of_representation
- responsiveness
- concession_and_common_ground
- respectful_engagement

Then select the overall better listener (Pro / Con / Tie) and optional justification/notes. Annotations are saved to `annotations.json`.

- **List all debates:** http://127.0.0.1:5000/list  
- **Previous / Next** links move through the valid debate set.

## Claude: try one debate (iterate on prompt)

Edit the prompts in:

- `prompt_templates/listening_system.txt` (system instructions/rubric)
- `prompt_templates/listening_user.txt` (debate payload + JSON output contract; placeholders: `{debate_topic}`, `{debate_text}`)

Then:

```bash
python try_prompt.py                  # uses first valid debate
python try_prompt.py --debate-id 358  # use a specific debate
python try_prompt.py --debate-ids 358,412,900  # multiple debates
python try_prompt.py --debate-ids "358 412 900" --save-results
python try_prompt.py --debate-ids 358,412,900 --max-workers 3 --max-retries 10 --retry-wait 15
python try_prompt.py --debate-ids 358,412,900 --save-results --enable-cache
```

This prints the prompt (truncated), sends it to Claude, and prints the raw response plus parsed structured JSON (including per-dimension scores and overall judgment).
When `--save-results` is provided, results are written to `claude_listening_trial.json` by default (override with `--output`).

## Claude: run on all debates

```bash
python run_claude_batch.py            # all valid debates
python run_claude_batch.py --limit 10 # first 10 only (for testing)
python run_claude_batch.py --model claude-3-5-sonnet-20241022
python run_claude_batch.py --use-batch --batch-poll-secs 30
python run_claude_batch.py --use-batch --enable-cache --cache-ttl 1h
```

Results are written to `claude_listening.json` with fields including:

- `debate_id`
- `judgment` / `overall_judgment`
- `evaluation` (full rubric JSON for Pro/Con + overall_better_listener)
- `model`
- `timestamp`
- `raw_response`
- `request_metadata` (request_id, temperature, max_tokens, prompt hashes, cache settings)
- `response_metadata` (message_id, stop_reason, latency if available)
- `usage` (input/output tokens + cache read/write token counts when provided)

By default, new runs append to existing output files to allow multiple annotations per debate.

Batch runs write progress reports and checkpoints under `reports/` (JSON). If a run fails mid-way, you can inspect the report and checkpoint to see how far it got.

Prompt caching:
- Use `--enable-cache` to mark a cached prefix in the user prompt.
- By default it looks for `[[CACHE_BREAK]]` in the user prompt template; everything before it is cached.
- If the marker is absent, the entire user prompt is marked cached (harmless but may reduce cache hit rate).

## Compare human vs Claude (RQ0)

After you have both annotations and Claude output:

```bash
python compare.py
python compare.py --report report.txt  # save report to file
python compare.py --use-trials         # compare against trial outputs
```

Prints overlap count, overall judgment agreement/accuracy, a confusion matrix (human vs Claude), and dimension-level MAE where both sides have numeric scores. If there are multiple annotations per debate (human or model), the comparison uses all pairwise combinations for those debates.

## RQ1 / RQ2 preparation

- **Outcomes:** The data loader exposes `get_debate_outcomes(debate_id)` (majority_winner, num_votes, num_flipped, pct_switched_to_pro/con, etc.) for correlating listening with persuasive success later.
- **Voter-level:** `get_votes_for_debate(debate_id)` returns voter_id, agreed_before, agreed_after, flipped for side-switching analysis (RQ2).

No extra scripts for RQ1/RQ2 in this repo; use the data layer and the saved JSON (annotations, claude_listening) when you’re ready.
