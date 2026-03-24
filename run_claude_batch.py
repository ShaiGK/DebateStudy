"""
Run Claude on all valid debates with the listening prompt. Saves to claude_listening.json.
Usage: python run_claude_batch.py [--limit N] [--model MODEL] [--use-batch]
"""
import argparse
import json
import os
import re
import time
import hashlib
import uuid
from datetime import datetime

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import anthropic

from config import (
    CLAUDE_LISTENING_PATH,
    LISTENING_SYSTEM_PROMPT_PATH,
    LISTENING_USER_PROMPT_PATH,
    REPORTS_DIR,
)
from data_loader import format_debate_text_for_prompt, get_debate, get_valid_debate_ids

LISTENING_DIMENSIONS = (
    "acknowledgment",
    "accuracy_of_representation",
    "responsiveness",
    "concession_and_common_ground",
    "respectful_engagement",
)
VALID_JUDGMENTS = {"Pro", "Con", "Tie"}


def load_prompt_templates() -> tuple[str, str]:
    if not LISTENING_SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"System prompt template not found: {LISTENING_SYSTEM_PROMPT_PATH}")
    if not LISTENING_USER_PROMPT_PATH.exists():
        raise FileNotFoundError(f"User prompt template not found: {LISTENING_USER_PROMPT_PATH}")
    return (
        LISTENING_SYSTEM_PROMPT_PATH.read_text(),
        LISTENING_USER_PROMPT_PATH.read_text(),
    )


def _empty_evaluation() -> dict:
    evaluation = {}
    for side in ("pro", "con"):
        evaluation[side] = {
            d: {"score": None, "justification": ""} for d in LISTENING_DIMENSIONS
        }
    evaluation["overall_better_listener"] = {"judgment": "Tie", "justification": ""}
    return evaluation


def _extract_json_block(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_structured_response(response_text: str) -> dict:
    """Parse and normalize Claude's JSON output for the listening rubric."""
    evaluation = _empty_evaluation()
    json_text = _extract_json_block(response_text)

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return evaluation

    for side in ("pro", "con"):
        side_obj = payload.get(side, {})
        if not isinstance(side_obj, dict):
            continue
        for dim in LISTENING_DIMENSIONS:
            dim_obj = side_obj.get(dim, {})
            if not isinstance(dim_obj, dict):
                continue
            score = dim_obj.get("score")
            normalized_score = None
            if isinstance(score, int) and 1 <= score <= 5:
                normalized_score = score
            elif isinstance(score, str) and score.isdigit() and 1 <= int(score) <= 5:
                normalized_score = int(score)

            evaluation[side][dim] = {
                "score": normalized_score,
                "justification": str(dim_obj.get("justification", "")).strip(),
            }

    overall = payload.get("overall_better_listener", {})
    if isinstance(overall, dict):
        judgment = str(overall.get("judgment", "Tie")).strip()
        if judgment.lower() == "pro":
            judgment = "Pro"
        elif judgment.lower() == "con":
            judgment = "Con"
        elif judgment.lower() == "tie":
            judgment = "Tie"
        else:
            judgment = "Tie"

        evaluation["overall_better_listener"] = {
            "judgment": judgment if judgment in VALID_JUDGMENTS else "Tie",
            "justification": str(overall.get("justification", "")).strip(),
        }

    return evaluation


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_cache_prefix(user_content: str, cache_marker: str) -> tuple[str, str]:
    if cache_marker and cache_marker in user_content:
        return user_content.split(cache_marker, 1)
    return user_content, ""


def _extract_usage(msg) -> dict:
    usage = getattr(msg, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        get = usage.get
    else:
        get = lambda k: getattr(usage, k, None)
    return {
        "input_tokens": get("input_tokens"),
        "output_tokens": get("output_tokens"),
        "cache_creation_input_tokens": get("cache_creation_input_tokens"),
        "cache_read_input_tokens": get("cache_read_input_tokens"),
    }


def _extract_response_meta(msg, latency_ms: int | None) -> dict:
    return {
        "message_id": getattr(msg, "id", None),
        "model": getattr(msg, "model", None),
        "stop_reason": getattr(msg, "stop_reason", None),
        "latency_ms": latency_ms,
    }


def _build_system_content_blocks(
    system_prompt: str,
    enable_cache: bool,
    cache_ttl: str | None,
):
    if not enable_cache:
        return system_prompt
    cache_control = {"type": "ephemeral"}
    if cache_ttl:
        cache_control["ttl"] = cache_ttl
    return [{"type": "text", "text": system_prompt, "cache_control": cache_control}]


def main():
    parser = argparse.ArgumentParser(description="Run Claude listening task on debates")
    parser.add_argument("--limit", type=int, default=None, help="Max number of debates to run")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6", help="Claude model")
    parser.add_argument("--use-batch", action="store_true", help="Use Claude message batches (discounted, slower)")
    parser.add_argument("--batch-poll-secs", type=int, default=30, help="Polling interval for batch results")
    parser.add_argument("--enable-cache", action="store_true", help="Enable Anthropic prompt caching")
    parser.add_argument("--cache-ttl", type=str, default=None, help="Cache TTL, e.g., 5m or 1h")
    parser.add_argument("--cache-marker", type=str, default="[[CACHE_BREAK]]", help="Marker to split cached prefix")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY in the environment or .env file.")
        return 1

    system_prompt, user_template = load_prompt_templates()
    system_prompt_hash = _hash_text(system_prompt)
    user_template_hash = _hash_text(user_template)
    system_content = _build_system_content_blocks(
        system_prompt, args.enable_cache, args.cache_ttl
    )
    ids = get_valid_debate_ids()
    if args.limit:
        ids = ids[: args.limit]
    print(f"Running on {len(ids)} debates (model: {args.model})")

    client = anthropic.Anthropic()
    results = []
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"run_claude_batch_{run_id}.json"
    checkpoint_path = REPORTS_DIR / f"run_claude_batch_{run_id}_checkpoint.json"
    cache_warning_path = REPORTS_DIR / f"cache_warning_run_{run_id}.txt"
    cache_warning_written = False

    def _write_report(status: str, details: dict) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "details": details,
        }
        report_path.write_text(json.dumps(payload, indent=2))

    def _write_checkpoint() -> None:
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "count": len(results),
            "results": results,
        }
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2))

    def _maybe_warn_cache(request_metadata: dict, usage: dict) -> None:
        nonlocal cache_warning_written
        if cache_warning_written or not args.enable_cache:
            return
        marker_found = request_metadata.get("cache_marker_found")
        if marker_found is False:
            reason = "cache marker not found in user prompt"
        else:
            cache_create = usage.get("cache_creation_input_tokens")
            cache_read = usage.get("cache_read_input_tokens")
            if cache_create is None or cache_read is None:
                reason = "usage missing cache token fields"
            elif cache_create == 0 and cache_read == 0:
                reason = "cache token fields are zero"
            else:
                return
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        cache_warning_path.write_text(
            "Cache warning:\n"
            f"reason: {reason}\n"
            f"cache_marker_found: {marker_found}\n"
            f"cached_prefix_chars: {request_metadata.get('cached_prefix_chars')}\n"
            f"cached_suffix_chars: {request_metadata.get('cached_suffix_chars')}\n"
            "Likely causes: prompt prefix too short, caching not enabled for the model/account, "
            "or cache TTL expired.\n"
        )
        print(f"Cache warning written to {cache_warning_path}")
        cache_warning_written = True

    def _build_user_content_blocks(user_content: str):
        if not args.enable_cache:
            return user_content
        if args.cache_marker and args.cache_marker in user_content:
            prefix, suffix = user_content.split(args.cache_marker, 1)
        else:
            prefix, suffix = user_content, ""
        cache_control = {"type": "ephemeral"}
        if args.cache_ttl:
            cache_control["ttl"] = args.cache_ttl
        blocks = [{"type": "text", "text": prefix, "cache_control": cache_control}]
        if suffix:
            blocks.append({"type": "text", "text": suffix})
        return blocks

    def _prime_cache(debate_id: int) -> None:
        debate = get_debate(debate_id)
        if not debate:
            print(f"Prime debate {debate_id} not found.")
            return
        debate_text = format_debate_text_for_prompt(debate)
        user_content = user_template.format(
            debate_topic=debate["proposition"],
            debate_text=debate_text,
        )
        user_blocks = _build_user_content_blocks(user_content)
        cache_marker_found = bool(args.cache_marker and args.cache_marker in user_content)
        print(f"Priming cache with debate {debate_id}.")
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        msg = client.messages.create(
            model=args.model,
            max_tokens=1200,
            temperature=0,
            system=system_content,
            messages=[{"role": "user", "content": user_blocks}],
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        response_text = msg.content[0].text if msg.content else ""
        evaluation = parse_structured_response(response_text)
        judgment = evaluation["overall_better_listener"]["judgment"]
        prefix, suffix = _split_cache_prefix(user_content, args.cache_marker)
        request_metadata = {
            "request_id": request_id,
            "model": args.model,
            "temperature": 0,
            "max_tokens": 1200,
            "cache_enabled": args.enable_cache,
            "cache_ttl": args.cache_ttl,
            "cache_marker": args.cache_marker,
            "cache_marker_found": cache_marker_found,
            "system_cached": bool(args.enable_cache),
            "system_prompt_hash": system_prompt_hash,
            "user_template_hash": user_template_hash,
            "user_prompt_hash": _hash_text(user_content),
            "cached_prefix_hash": _hash_text(prefix),
            "cached_suffix_hash": _hash_text(suffix),
            "cached_prefix_chars": len(prefix),
            "cached_suffix_chars": len(suffix),
        }
        response_metadata = _extract_response_meta(msg, latency_ms)
        usage = _extract_usage(msg)
        results.append(
            {
                "debate_id": debate_id,
                "judgment": judgment,
                "overall_judgment": judgment,
                "evaluation": evaluation,
                "model": args.model,
                "timestamp": datetime.now().isoformat(),
                "request_metadata": request_metadata,
                "response_metadata": response_metadata,
                "usage": usage,
            }
        )
        _maybe_warn_cache(request_metadata, usage)
        _write_checkpoint()

    try:
        if not args.use_batch:
            _write_report(
                "running",
                {
                    "mode": "standard",
                    "model": args.model,
                    "total": len(ids),
                },
            )
            for i, debate_id in enumerate(ids, 1):
                debate = get_debate(debate_id)
                if not debate:
                    continue
                debate_text = format_debate_text_for_prompt(debate)
                user_content = user_template.format(
                    debate_topic=debate["proposition"],
                    debate_text=debate_text,
                )
                user_blocks = _build_user_content_blocks(user_content)
                try:
                    request_id = str(uuid.uuid4())
                    start = time.perf_counter()
                    msg = client.messages.create(
                        model=args.model,
                        max_tokens=1200,
                        temperature=0,
                        system=system_content,
                        messages=[{"role": "user", "content": user_blocks}],
                    )
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    response_text = msg.content[0].text if msg.content else ""
                    evaluation = parse_structured_response(response_text)
                    judgment = evaluation["overall_better_listener"]["judgment"]
                    cache_marker_found = bool(args.cache_marker and args.cache_marker in user_content)
                    prefix, suffix = _split_cache_prefix(user_content, args.cache_marker)
                    request_metadata = {
                        "request_id": request_id,
                        "model": args.model,
                        "temperature": 0,
                        "max_tokens": 1200,
                        "cache_enabled": args.enable_cache,
                        "cache_ttl": args.cache_ttl,
                        "cache_marker": args.cache_marker,
                        "cache_marker_found": cache_marker_found,
                        "system_cached": bool(args.enable_cache),
                        "system_prompt_hash": system_prompt_hash,
                        "user_template_hash": user_template_hash,
                        "user_prompt_hash": _hash_text(user_content),
                        "cached_prefix_hash": _hash_text(prefix),
                        "cached_suffix_hash": _hash_text(suffix),
                        "cached_prefix_chars": len(prefix),
                        "cached_suffix_chars": len(suffix),
                    }
                    response_metadata = _extract_response_meta(msg, latency_ms)
                    usage = _extract_usage(msg)
                except Exception as e:
                    print(f"  Debate {debate_id} error: {e}")
                    response_text = str(e)
                    evaluation = _empty_evaluation()
                    judgment = "Tie"
                    cache_marker_found = bool(args.cache_marker and args.cache_marker in user_content)
                    request_metadata = {
                        "request_id": request_id,
                        "model": args.model,
                        "temperature": 0,
                        "max_tokens": 1200,
                        "cache_enabled": args.enable_cache,
                        "cache_ttl": args.cache_ttl,
                        "cache_marker": args.cache_marker,
                        "cache_marker_found": cache_marker_found,
                        "system_cached": bool(args.enable_cache),
                        "system_prompt_hash": system_prompt_hash,
                        "user_template_hash": user_template_hash,
                        "user_prompt_hash": _hash_text(user_content),
                    }
                    response_metadata = {"error": str(e)}
                    usage = {}

                results.append(
                    {
                        "debate_id": debate_id,
                        "judgment": judgment,
                        "overall_judgment": judgment,
                        "evaluation": evaluation,
                        "model": args.model,
                        "timestamp": datetime.now().isoformat(),
                        "request_metadata": request_metadata,
                        "response_metadata": response_metadata,
                        "usage": usage,
                    }
                )
                _maybe_warn_cache(request_metadata, usage)
                if i % 50 == 0:
                    print(f"  Done {i}/{len(ids)}")
                    _write_checkpoint()

            _write_report(
                "completed",
                {
                    "mode": "standard",
                    "model": args.model,
                    "total": len(ids),
                    "completed": len(results),
                },
            )
        else:
            if not hasattr(client.messages, "batches"):
                raise RuntimeError(
                    "Anthropic SDK does not support message batches. "
                    "Upgrade anthropic or run without --use-batch."
                )

            batch_ids = list(ids)
            if args.enable_cache and len(batch_ids) > 1:
                prime_id = batch_ids.pop(0)
                try:
                    _prime_cache(prime_id)
                except Exception as e:
                    print(f"Prime failed for debate {prime_id}: {e}. Continuing without primed cache.")
                    batch_ids.insert(0, prime_id)
                    _write_report(
                        "prime_failed",
                        {
                            "mode": "batch",
                            "model": args.model,
                            "total": len(ids),
                            "primed_debate_id": prime_id,
                            "error": str(e),
                            "remaining": len(batch_ids),
                        },
                    )
                else:
                    _write_report(
                        "primed",
                        {
                            "mode": "batch",
                            "model": args.model,
                            "total": len(ids),
                            "primed_debate_id": prime_id,
                            "remaining": len(batch_ids),
                        },
                    )

            requests = []
            debate_map = {}
            request_meta_map = {}
            for debate_id in batch_ids:
                debate = get_debate(debate_id)
                if not debate:
                    continue
                debate_text = format_debate_text_for_prompt(debate)
                user_content = user_template.format(
                    debate_topic=debate["proposition"],
                    debate_text=debate_text,
                )
                user_blocks = _build_user_content_blocks(user_content)
                request_id = str(uuid.uuid4())
                prefix, suffix = _split_cache_prefix(user_content, args.cache_marker)
                cache_marker_found = bool(args.cache_marker and args.cache_marker in user_content)
                request_meta_map[str(debate_id)] = {
                    "request_id": request_id,
                    "model": args.model,
                    "temperature": 0,
                    "max_tokens": 1200,
                    "cache_enabled": args.enable_cache,
                    "cache_ttl": args.cache_ttl,
                    "cache_marker": args.cache_marker,
                    "cache_marker_found": cache_marker_found,
                    "system_cached": bool(args.enable_cache),
                    "system_prompt_hash": system_prompt_hash,
                    "user_template_hash": user_template_hash,
                    "user_prompt_hash": _hash_text(user_content),
                    "cached_prefix_hash": _hash_text(prefix),
                    "cached_suffix_hash": _hash_text(suffix),
                    "cached_prefix_chars": len(prefix),
                    "cached_suffix_chars": len(suffix),
                }
                custom_id = str(debate_id)
                debate_map[custom_id] = debate_id
                requests.append(
                    {
                        "custom_id": custom_id,
                        "params": {
                            "model": args.model,
                            "max_tokens": 1200,
                            "temperature": 0,
                            "system": system_content,
                            "messages": [{"role": "user", "content": user_blocks}],
                        },
                    }
                )

            if not requests:
                _write_report(
                    "completed",
                    {
                        "mode": "batch",
                        "model": args.model,
                        "total": len(ids),
                        "completed": len(results),
                        "note": "No remaining requests after priming.",
                    },
                )
                CLAUDE_LISTENING_PATH.write_text(json.dumps(results, indent=2))
                print(f"Saved to {CLAUDE_LISTENING_PATH}")
                return 0

            print(f"Submitting batch with {len(requests)} requests.")
            batch = client.messages.batches.create(requests=requests)
            batch_id = batch.id if hasattr(batch, "id") else batch.get("id")
            print(f"Batch created: {batch_id}")
            _write_report(
                "submitted",
                {
                    "mode": "batch",
                    "model": args.model,
                    "total": len(requests),
                    "batch_id": batch_id,
                },
            )

            status = None
            while True:
                batch = client.messages.batches.retrieve(batch_id)
                status = (
                    batch.processing_status
                    if hasattr(batch, "processing_status")
                    else batch.get("processing_status")
                )
                if status in {"completed", "failed", "canceled", "expired"}:
                    break
                print(f"  Batch status: {status}. Waiting {args.batch_poll_secs}s.")
                time.sleep(args.batch_poll_secs)

            if status != "completed":
                _write_report(
                    "ended",
                    {
                        "mode": "batch",
                        "model": args.model,
                        "total": len(requests),
                        "batch_id": batch_id,
                        "status": status,
                        "completed": len(results),
                    },
                )
                raise RuntimeError(f"Batch ended with status: {status}")

            result_items = client.messages.batches.list_results(batch_id)
            if hasattr(result_items, "data"):
                result_items = result_items.data

            for item in result_items:
                custom_id = (
                    item.custom_id
                    if hasattr(item, "custom_id")
                    else item.get("custom_id")
                )
                result = item.result if hasattr(item, "result") else item.get("result", {})
                debate_id = debate_map.get(str(custom_id), custom_id)
                request_metadata = request_meta_map.get(str(debate_id), {}).copy()
                request_metadata["batch_id"] = batch_id
                request_metadata["custom_id"] = custom_id

                if result and (
                    result.get("type") == "succeeded" or hasattr(result, "message")
                ):
                    message = (
                        result.get("message")
                        if isinstance(result, dict)
                        else result.message
                    )
                    content = (
                        message.get("content")
                        if isinstance(message, dict)
                        else message.content
                    )
                    response_text = content[0].text if content else ""
                    evaluation = parse_structured_response(response_text)
                    judgment = evaluation["overall_better_listener"]["judgment"]
                    response_metadata = _extract_response_meta(message, None)
                    usage = _extract_usage(message)
                else:
                    error = result.get("error") if isinstance(result, dict) else None
                    response_text = str(error) if error else "Batch request failed."
                    evaluation = _empty_evaluation()
                    judgment = "Tie"
                    response_metadata = {"error": response_text}
                    usage = {}

                results.append(
                    {
                        "debate_id": int(debate_id),
                        "judgment": judgment,
                        "overall_judgment": judgment,
                        "evaluation": evaluation,
                        "model": args.model,
                        "timestamp": datetime.now().isoformat(),
                        "request_metadata": request_metadata,
                        "response_metadata": response_metadata,
                        "usage": usage,
                    }
                )
                _maybe_warn_cache(request_metadata, usage)
                if len(results) % 50 == 0:
                    _write_checkpoint()

            _write_report(
                "completed",
                {
                    "mode": "batch",
                    "model": args.model,
                    "total": len(requests),
                    "batch_id": batch_id,
                    "completed": len(results),
                },
            )

        if CLAUDE_LISTENING_PATH.exists():
            try:
                existing = json.loads(CLAUDE_LISTENING_PATH.read_text())
            except json.JSONDecodeError:
                existing = []
            if not isinstance(existing, list):
                existing = []
            results = existing + results
        CLAUDE_LISTENING_PATH.write_text(json.dumps(results, indent=2))
        print(f"Saved to {CLAUDE_LISTENING_PATH}")
        return 0
    except Exception as e:
        _write_report(
            "failed",
            {
                "mode": "batch" if args.use_batch else "standard",
                "model": args.model,
                "total": len(ids),
                "completed": len(results),
                "error": str(e),
            },
        )
        if results:
            _write_checkpoint()
        raise


if __name__ == "__main__":
    exit(main())
