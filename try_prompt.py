"""
Try the listening prompt on one or more debates. Useful for iterating on prompt wording (RQ0).
Usage: python try_prompt.py [--debate-id ID] [--debate-ids 1,2,3] [--save-results]
"""
import argparse
import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from anthropic import Anthropic

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import anthropic

from config import CLAUDE_LISTENING_TRIAL_PATH, REPORTS_DIR
from data_loader import get_valid_debate_ids, get_debate, format_debate_text_for_prompt
from run_claude_batch import load_prompt_templates, parse_structured_response


def _parse_debate_ids(raw: str) -> list[int]:
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]
    ids = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            ids.append(int(part))
        else:
            raise ValueError(f"Invalid debate id: {part}")
    return ids


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
            "rate limit" in msg
            or "rate_limit" in msg
            or "429" in msg
            or exc.__class__.__name__.lower() in {"ratelimiterror", "rate_limit_error"}
    )


def _call_with_retries(
        *,
        model: str,
        system_content,
        user_content,
        user_content_raw: str,
        max_retries: int,
        retry_wait: int,
        log_path: str | None,
        debate_id: int,
        request_id: str,
) -> str:
    attempt = 0
    while True:
        try:
            client = anthropic.Anthropic()
            start = time.perf_counter()
            msg = client.messages.create(
                model=model,
                max_tokens=1200,
                temperature=0,
                system=system_content,
                messages=[{"role": "user", "content": user_content}],
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            response_text = msg.content[0].text if msg.content else ""
            return response_text, msg, latency_ms, request_id, user_content_raw
        except Exception as e:
            if attempt >= max_retries or not _is_rate_limited(e):
                raise
            attempt += 1
            if log_path:
                _log_retry(
                    log_path,
                    debate_id=debate_id,
                    attempt=attempt,
                    wait_seconds=retry_wait,
                    error=str(e),
                )
            time.sleep(retry_wait)


def _log_retry(
        log_path: str,
        *,
        debate_id: int,
        attempt: int,
        wait_seconds: int,
        error: str,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now().isoformat(),
        "event": "rate_limit_retry",
        "debate_id": debate_id,
        "attempt": attempt,
        "wait_seconds": wait_seconds,
        "error": error,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def _build_user_content_blocks(
        user_content: str,
        enable_cache: bool,
        cache_marker: str,
        cache_ttl: str | None,
):
    if not enable_cache:
        return user_content

    if cache_marker and cache_marker in user_content:
        prefix, suffix = user_content.split(cache_marker, 1)
    else:
        prefix, suffix = user_content, ""

    cache_control = {"type": "ephemeral"}
    if cache_ttl:
        cache_control["ttl"] = cache_ttl

    blocks = [{"type": "text", "text": prefix, "cache_control": cache_control}]
    if suffix:
        blocks.append({"type": "text", "text": suffix})
    return blocks


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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_token_number(
        *,
        model: str,
        system_prompt: str,
        user_content: str,
        cache_marker: str,
        cache_ttl: str | None,
) -> int:
    """
    Return the official token count for the cacheable portion (system + cached user prefix).
    This uses Anthropic's count_tokens on the exact cached blocks.
    """
    system_content = _build_system_content_blocks(system_prompt, True, cache_ttl)
    prefix, _ = _split_cache_prefix(user_content, cache_marker)
    user_blocks = _build_user_content_blocks(
        user_content=prefix,
        enable_cache=True,
        cache_marker="",
        cache_ttl=cache_ttl,
    )

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    token_count = client.messages.count_tokens(
        model=model,
        system=system_content,
        messages=[{"role": "user", "content": user_blocks}],
    ).input_tokens
    print(token_count)
    return token_count


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debate-id", type=int, default=None, help="Debate ID (default: first valid)")
    parser.add_argument("--debate-ids", type=str, default=None, help="Comma-separated debate IDs")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--save-results", action="store_true", help="Save trial results to a JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output path for saved results")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel requests for trial runs")
    parser.add_argument("--max-retries", type=int, default=10, help="Retries on 429 rate limits")
    parser.add_argument("--retry-wait", type=int, default=15, help="Seconds between rate-limit retries")
    parser.add_argument("--enable-cache", action="store_true", help="Enable Anthropic prompt caching")
    parser.add_argument("--cache-ttl", type=str, default=None, help="Cache TTL, e.g., 5m or 1h")
    parser.add_argument("--cache-marker", type=str, default="[[CACHE_BREAK]]", help="Marker to split cached prefix")
    parser.add_argument("--verify-cache-tokens", action="store_true", help="Print cacheable token count and exit")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY in the environment or .env file.")
        return 1

    ids = get_valid_debate_ids()
    if not ids:
        print("No valid debates.")
        return 1
    try:
        if args.debate_ids:
            debate_ids = _parse_debate_ids(args.debate_ids)
        elif args.debate_id is not None:
            debate_ids = [args.debate_id]
        else:
            debate_ids = [ids[0]]
    except ValueError as e:
        print(str(e))
        return 1

    invalid = [d for d in debate_ids if d not in ids]
    if invalid:
        print(f"Debate IDs not in valid set: {invalid}. First valid: {ids[0]}")
        return 1

    system_prompt, user_template = load_prompt_templates()
    system_prompt_hash = _hash_text(system_prompt)
    user_template_hash = _hash_text(user_template)
    system_content = _build_system_content_blocks(
        system_prompt, args.enable_cache, args.cache_ttl
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_warning_path = REPORTS_DIR / f"cache_warning_try_prompt_{run_id}.txt"
    cache_warning_written = False
    retry_log_path = str(
        REPORTS_DIR / f"try_prompt_retries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    print("--- System prompt (full) ---")
    print(system_prompt)
    if args.verify_cache_tokens:
        debate_id = debate_ids[0]
        debate = get_debate(debate_id)
        if not debate:
            print("Debate not found.")
            return 1
        debate_text = format_debate_text_for_prompt(debate)
        user_content = user_template.format(
            debate_topic=debate["proposition"],
            debate_text=debate_text,
        )
        verify_token_number(
            model=args.model,
            system_prompt=system_prompt,
            user_content=user_content,
            cache_marker=args.cache_marker,
            cache_ttl=args.cache_ttl,
        )
        return 0

    print(f"\n--- Running {len(debate_ids)} debate(s) ---")

    results = []

    def _print_and_store(
            *,
            debate_id: int,
            response_text: str,
            evaluation: dict,
            judgment: str,
            user_prompt_preview: str | None = None,
            user_prompt_truncated: bool = False,
            request_metadata: dict | None = None,
            response_metadata: dict | None = None,
            usage: dict | None = None,
    ) -> None:
        if user_prompt_preview is not None:
            print(f"\n--- Debate {debate_id} user prompt (first 1500 chars) ---")
            print(user_prompt_preview)
            if user_prompt_truncated:
                print("...")
            print("--- Sending to Claude ---")

        print("Raw response:", repr(response_text))
        print("Parsed evaluation:")
        print(json.dumps(evaluation, indent=2))
        print("Parsed overall judgment:", judgment)
        print("Usage:", usage or {})

        results.append(
            {
                "debate_id": debate_id,
                "judgment": judgment,
                "overall_judgment": judgment,
                "evaluation": evaluation,
                "model": args.model,
                "timestamp": datetime.now().isoformat(),
                "trial": True,
                "request_metadata": request_metadata or {},
                "response_metadata": response_metadata or {},
                "usage": usage or {},
            }
        )

        _maybe_warn_cache(request_metadata or {}, usage or {})

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

    def _run_one(debate_id: int) -> dict:
        debate = get_debate(debate_id)
        if not debate:
            return {
                "debate_id": debate_id,
                "error": "Debate not found.",
            }
        debate_text = format_debate_text_for_prompt(debate)
        user_content = user_template.format(
            debate_topic=debate["proposition"],
            debate_text=debate_text,
        )
        user_blocks = _build_user_content_blocks(
            user_content=user_content,
            enable_cache=args.enable_cache,
            cache_marker=args.cache_marker,
            cache_ttl=args.cache_ttl,
        )
        cache_marker_found = bool(args.cache_marker and args.cache_marker in user_content)
        request_id = str(uuid.uuid4())
        response_text, msg, latency_ms, request_id, user_content_raw = _call_with_retries(
            model=args.model,
            system_content=system_content,
            user_content=user_blocks,
            user_content_raw=user_content,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
            log_path=retry_log_path,
            debate_id=debate_id,
            request_id=request_id,
        )
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
            "user_prompt_hash": _hash_text(user_content_raw),
            "cached_prefix_hash": _hash_text(prefix),
            "cached_suffix_hash": _hash_text(suffix),
            "cached_prefix_chars": len(prefix),
            "cached_suffix_chars": len(suffix),
        }
        response_metadata = _extract_response_meta(msg, latency_ms)
        usage = _extract_usage(msg)
        return {
            "debate_id": debate_id,
            "user_prompt_preview": user_content[:1500],
            "user_prompt_truncated": len(user_content) > 1500,
            "response_text": response_text,
            "evaluation": evaluation,
            "judgment": judgment,
            "request_metadata": request_metadata,
            "response_metadata": response_metadata,
            "usage": usage,
        }

    remaining_ids = list(debate_ids)
    if args.enable_cache and len(remaining_ids) > 1:
        prime_id = remaining_ids.pop(0)
        try:
            out = _run_one(prime_id)
        except Exception as e:
            print(f"  Debate {prime_id} error: {e}")
            response_text = str(e)
            evaluation = parse_structured_response("")
            judgment = "Tie"
            _print_and_store(
                debate_id=prime_id,
                response_text=response_text,
                evaluation=evaluation,
                judgment=judgment,
                request_metadata=None,
                response_metadata=None,
                usage=None,
            )
        else:
            if "error" in out:
                print(f"  Debate {prime_id} error: {out['error']}")
                response_text = out["error"]
                evaluation = parse_structured_response("")
                judgment = "Tie"
                _print_and_store(
                    debate_id=prime_id,
                    response_text=response_text,
                    evaluation=evaluation,
                    judgment=judgment,
                )
            else:
                _print_and_store(
                    debate_id=prime_id,
                    response_text=out["response_text"],
                    evaluation=out["evaluation"],
                    judgment=out["judgment"],
                    user_prompt_preview=out["user_prompt_preview"],
                    user_prompt_truncated=out["user_prompt_truncated"],
                    request_metadata=out.get("request_metadata"),
                    response_metadata=out.get("response_metadata"),
                    usage=out.get("usage"),
                )

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {pool.submit(_run_one, debate_id): debate_id for debate_id in remaining_ids}
        for fut in as_completed(futures):
            debate_id = futures[fut]
            user_prompt_preview = None
            user_prompt_truncated = False
            request_metadata = None
            response_metadata = None
            usage = None
            try:
                out = fut.result()
            except Exception as e:
                print(f"  Debate {debate_id} error: {e}")
                response_text = str(e)
                evaluation = parse_structured_response("")
                judgment = "Tie"
            else:
                if "error" in out:
                    print(f"  Debate {debate_id} error: {out['error']}")
                    response_text = out["error"]
                    evaluation = parse_structured_response("")
                    judgment = "Tie"
                else:
                    response_text = out["response_text"]
                    evaluation = out["evaluation"]
                    judgment = out["judgment"]
                    user_prompt_preview = out["user_prompt_preview"]
                    user_prompt_truncated = out["user_prompt_truncated"]
                    request_metadata = out.get("request_metadata")
                    response_metadata = out.get("response_metadata")
                    usage = out.get("usage")

            _print_and_store(
                debate_id=debate_id,
                response_text=response_text,
                evaluation=evaluation,
                judgment=judgment,
                user_prompt_preview=user_prompt_preview,
                user_prompt_truncated=user_prompt_truncated,
                request_metadata=request_metadata,
                response_metadata=response_metadata,
                usage=usage,
            )

    if args.save_results:
        output_path = args.output if args.output else str(CLAUDE_LISTENING_TRIAL_PATH)
        if os.path.isfile(output_path):
            with open(output_path) as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
            if not isinstance(existing, list):
                existing = []
            results = existing + results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved trial results to {output_path}")
    return 0


def count_tokens(did=None):
    ids = get_valid_debate_ids()
    if not ids:
        print("No valid debates.")
        return 1

    debate_id = did if did is not None else ids[0]
    if debate_id not in ids:
        print(f"Debate {debate_id} not in valid set. First valid: {ids[0]}")
        return 1

    debate = get_debate(debate_id)
    if not debate:
        print("Debate not found.")
        return 1

    system_prompt, user_template = load_prompt_templates()
    debate_text = format_debate_text_for_prompt(debate)
    user_content = user_template.format(
        debate_topic=debate["proposition"],
        debate_text=debate_text,
    )
    # user_content = user_template.format(
    #     debate_topic="",
    #     debate_text="",
    # )

    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
    )
    message_tokens_count = client.messages.count_tokens(
        model="claude-sonnet-4-6",
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    print(message_tokens_count.input_tokens)


if __name__ == "__main__":
    exit(main())
    # count_tokens(358)
    # count_tokens(556)
