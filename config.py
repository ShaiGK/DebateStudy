"""Paths and config for debate_study. No secrets in code."""
from pathlib import Path

# Project root: debate_study/
PROJECT_DIR = Path(__file__).resolve().parent
# Thesis root (parent of debate_study)
THESIS_ROOT = PROJECT_DIR.parent
# Data lives under Thesis/data/processing
DATA_DIR = THESIS_ROOT / "data" / "processing"

# Data files (read-only)
DEBATES_FILTERED_PATH = DATA_DIR / "filtered_data" / "debates_filtered_df.json"
ROUNDS_PATH = DATA_DIR / "processed_data" / "rounds_df.json"
PROPOSITIONS_PATH = DATA_DIR / "propositions" / "propositions.json"
VOTES_FILTERED_PATH = DATA_DIR / "filtered_data" / "votes_filtered_df.json"
USERS_PATH = DATA_DIR / "processed_data" / "users_df.json"

# Outputs (created at runtime)
ANNOTATIONS_PATH = PROJECT_DIR / "annotations.json"
CLAUDE_LISTENING_PATH = PROJECT_DIR / "claude_listening.json"
CLAUDE_LISTENING_TRIAL_PATH = PROJECT_DIR / "claude_listening_trial.json"
REPORTS_DIR = PROJECT_DIR / "reports"

# Prompt template
PROMPT_TEMPLATE_DIR = PROJECT_DIR / "prompt_templates"
LISTENING_SYSTEM_PROMPT_PATH = PROMPT_TEMPLATE_DIR / "listening_system.txt"
LISTENING_USER_PROMPT_PATH = PROMPT_TEMPLATE_DIR / "listening_user.txt"
