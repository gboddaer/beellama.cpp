#!/usr/bin/env python3
"""Vulkan gap-closure benchmark runner with persistent-server and provenance support."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Validation thresholds
MAX_TPS_SENTINEL = 10_000  # t/s above this is implausible
MIN_VALID_TOKENS = 2       # fewer tokens is not useful
MAX_PROMPT_ECHO_COUNT = 2  # more occurrences of instruction phrase = echo


def extract_measurement(response, prompt_kind="coding", mode="base"):
    """Extract measurement from response with validity checking.

    Args:
        response: API response JSON
        prompt_kind: "coding" or "math" for prompt-specific validation
        mode: "base", "mtp", or "dflash" for speculative-mode draft checks

    Returns dict with measurement fields including validity.
    """
    timings = response.get("timings") or {}
    draft_n = int(timings.get("draft_n") or 0)
    draft_n_accepted = int(timings.get("draft_n_accepted") or 0)

    # Extract content, finish reason, and tokens
    content = ""
    finish_reason = None
    tokens_predicted = 0

    if "choices" in response:
        # oaicompat format: /v1/completions, /v1/chat/completions
        choices = response.get("choices", [])
        if choices:
            choice = choices[0]
            content = choice.get("text", "") or choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason")
            tokens_predicted = response.get("usage", {}).get("completion_tokens", 0) or choice.get("tokens_predicted", 0)
    else:
        # non-oaicompat format: /completion
        content = response.get("content", "")
        tokens_predicted = int(response.get("tokens_predicted") or 0)
        stop_type = response.get("stop", False)
        finish_reason = "stop" if stop_type else "length"

    predicted_per_second = float(timings.get("predicted_per_second") or 0.0)

    # Calculate draft acceptance
    draft_accept_pct = (100.0 * draft_n_accepted / draft_n) if draft_n else None

    # Raw token IDs from response
    token_ids = [int(t) for t in response.get("tokens", [])] if response.get("tokens") else []

    # Validity checking
    invalid_reasons = []

    # Check for empty content
    if not content.strip():
        invalid_reasons.append("empty_content")

    # Check for too few tokens
    if tokens_predicted <= MIN_VALID_TOKENS:
        invalid_reasons.append("too_few_tokens")

    # Check for invalid throughput
    if predicted_per_second <= 0:
        invalid_reasons.append("invalid_tps")

    # Check for implausibly high throughput (sentinel values)
    if predicted_per_second >= MAX_TPS_SENTINEL:
        invalid_reasons.append("implausible_tps")

    # Check for prompt echo
    instruction_phrases = {
        "coding": ["Only output the code", "def fibonacci"],
        "math": ["quadratic equation", "2.4 hours"],
    }
    prompt_phrases = instruction_phrases.get(prompt_kind, [])
    for phrase in prompt_phrases:
        if content.count(phrase) > MAX_PROMPT_ECHO_COUNT:
            invalid_reasons.append("prompt_echo")
            break

    # Speculative mode checks: MTP and DFlash require draft tokens.
    if mode in ("mtp", "dflash") and draft_n == 0:
        invalid_reasons.append("spec_no_drafts")

    # Coding-specific checks
    if prompt_kind == "coding":
        if finish_reason != "stop":
            invalid_reasons.append("coding_not_stopped")

        # Try to extract and compile Python code
        python_code = _extract_python_code(content)
        if python_code:
            try:
                compile(python_code, "<model-output>", "exec")
                if "def fibonacci" not in python_code:
                    invalid_reasons.append("coding_not_python")
            except SyntaxError:
                invalid_reasons.append("coding_not_python")
        else:
            # No Python code block found in coding mode is invalid
            invalid_reasons.append("coding_not_python")

    # Math-specific checks
    if prompt_kind == "math":
        if "2.4" not in content.lower():
            invalid_reasons.append("math_wrong_answer")

    return {
        "tokens_predicted": tokens_predicted,
        "predicted_per_second": predicted_per_second,
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "draft_accept_pct": draft_accept_pct,
        "finish_reason": finish_reason,
        "content_sha256": hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest(),
        "content": content,
        "token_ids": token_ids,
        "tokens_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if token_ids else "",
        "valid": len(invalid_reasons) == 0,
        "invalid_reasons": invalid_reasons,
    }


def _extract_python_code(content):
    """Extract Python code from markdown fenced or plain response.

    Never raises for model output; returns remaining text when closing
    fence is absent.
    """
    # Try fenced python blocks first
    start_marker = content.find("```python")
    if start_marker >= 0:
        start = start_marker + 9
        end = content.find("```", start + 1)
        if end < 0:
            return content[start:].strip()
        return content[start:end].strip()
    # Try generic code blocks
    start_marker = content.find("```")
    if start_marker >= 0:
        start = start_marker + 3
        end = content.find("```", start + 1)
        if end < 0:
            return content[start:].strip()
        return content[start:end].strip()
    # No fenced block: attempt to compile the entire content as Python.
    try:
        compile(content, "<model-output>", "exec")
        return content
    except SyntaxError:
        return None


def make_provenance_record(server_path, model_path, draft_path=None, 
                          device_line="", command=None, environment=None,
                          reference_label=None, external_source_head=None):
    """Create a comprehensive provenance record.
    
    Args:
        reference_label: Optional label for external reference binaries (e.g., "reference-adb92").
        external_source_head: Optional explicit source HEAD for external binaries.
            When provided, overrides git rev-parse of the worktree.
    """
    record = {
        "server_path": str(server_path),
        "server_version": "",
        "server_sha256": sha256_file(server_path) if os.path.isfile(server_path) else "",
        "source_head": "",
        "reference_label": reference_label or "",
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path) if os.path.isfile(model_path) else "",
        "model_size": os.path.getsize(model_path) if os.path.isfile(model_path) else 0,
    }
    
    if draft_path:
        record["draft_path"] = str(draft_path)
        record["draft_sha256"] = sha256_file(draft_path) if os.path.isfile(draft_path) else ""
        record["draft_size"] = os.path.getsize(draft_path) if os.path.isfile(draft_path) else 0
    
    record["device"] = device_line
    record["command"] = command or []
    record["environment"] = environment or {}
    record["timestamp"] = time.time()
    
    # Try to get server version (--version writes to stderr in llama-server).
    try:
        ver = subprocess.run(
            [str(server_path), "--version"],
            check=True,
            text=True,
            capture_output=True,
        )
        record["server_version"] = (ver.stdout.strip() or ver.stderr.strip())
    except Exception:
        pass
    
    # Get source HEAD: use external override if provided, otherwise worktree git HEAD
    if external_source_head:
        record["source_head"] = external_source_head
    else:
        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True
            ).strip()
            record["source_head"] = head
        except Exception:
            pass
    
    return record


def summary_stats(values):
    """Calculate summary statistics."""
    if not values:
        return None
    med = statistics.median(values)
    deviations = [abs(value - med) for value in values]
    return {
        "median": med,
        "min": min(values),
        "max": max(values),
        "mad": statistics.median(deviations),
        "count": len(values),
    }


def sha256_file(path):
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def http_json(url, payload=None, timeout=600):
    """POST JSON payload to url, return parsed JSON. Raises on HTTP error."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def wait_ready(proc, port, timeout_s=180):
    """Wait for server to become healthy."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup: {proc.returncode}")
        try:
            http_json(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s")


def find_device(server):
    """Find a Vulkan device matching criteria."""
    text = subprocess.run(
        [str(server), "--list-devices"], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout
    for line in text.splitlines():
        if "NAVI31" in line or "GFX1151" in line:
            return line.split(":", 1)[0].strip(), line.strip()
    raise RuntimeError("no Vulkan NAVI31/GFX1151 device found")


def mode_args(mode, draft):
    """Generate mode-specific command line arguments."""
    if mode == "base":
        return ["--spec-type", "none"]
    if mode == "mtp":
        return ["--spec-type", "draft-mtp", "--spec-draft-n-max", "8"]
    if mode == "dflash":
        args_list = [
            "--spec-type", "dflash",
            "--spec-draft-model", str(draft),
            "--spec-draft-n-max", "8",
        ]
        return args_list
    raise ValueError(f"Unknown mode: {mode}")


def build_server_command(server_path, model_path, draft_path, port, n_parallel,
                         device_id, mode, batch_size, ubatch_size, ctx_size):
    """Build the server command line list."""
    common = [
        str(server_path), "-m", str(model_path),
        "--port", str(port), "-np", str(n_parallel),
        "--kv-unified", "-ngl", "all",
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "--ctx-size", str(ctx_size),
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--cache-ram", "0",
        "--flash-attn", "on",
        "--device", str(device_id),
        "--reasoning", "off",
        "--no-mmap", "--no-host", "--host", "127.0.0.1",
    ]
    if mode == "base":
        spec = ["--spec-type", "none"]
    elif mode == "mtp":
        spec = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "8"]
    elif mode == "dflash":
        spec = [
            "--spec-type", "dflash",
            "--spec-draft-model", str(draft_path),
            "--spec-draft-n-max", "8",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return common + spec


def expected_source_head(external_source_head, worktree_head):
    """Return the expected source HEAD: external override or worktree."""
    return external_source_head or worktree_head


def open_output_exclusive(path):
    """Open file for exclusive creation; raises FileExistsError if it exists."""
    return open(path, "x", encoding="utf-8")


def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Vulkan gap-closure benchmark runner."
    )
    parser.add_argument("mode", choices=["base", "mtp", "dflash"])
    parser.add_argument("prompt", choices=["coding", "math"])
    parser.add_argument(
        "--repetitions", type=int, default=5,
        help="Number of measured repetitions (default: 5).",
    )
    parser.add_argument(
        "--gen-tokens", type=int, default=512,
        help="Number of tokens to generate per request (default: 512).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--warmup-tokens", type=int, default=32,
        help="Tokens for the warm-up request (default: 32).",
    )
    # Persistent-server options
    parser.add_argument(
        "--restart-between-reps", action="store_true",
        help="Restart server between repetitions (default: false).",
    )
    parser.add_argument(
        "--skip-warmup", action="store_true",
        help="Skip warm-up request (default: false).",
    )
    # External reference binary options
    parser.add_argument(
        "--reference-label", type=str, default=None,
        help="Label for external reference binary (e.g., 'reference-adb92').",
    )
    parser.add_argument(
        "--external-source-head", type=str, default=None,
        help="Explicit source HEAD for external reference binary.",
    )
    args = parser.parse_args()

    # --- configuration from environment ---
    server_path = os.environ.get("VK_GAP_SERVER", str(ROOT / "build" / "bin" / "llama-server"))
    model_path = os.environ.get("VK_GAP_MODEL")
    draft_path = os.environ.get("VK_GAP_DRAFT")
    port = int(os.environ.get("VK_GAP_PORT", "8099"))
    temp = float(os.environ.get("VK_GAP_TEMP", "0"))
    top_k = int(os.environ.get("VK_GAP_TOP_K", "20"))
    np = os.environ.get("VK_GAP_NP", "1")

    # --- pre-flight checks ---
    if not os.path.isfile(server_path):
        print(f"missing server: {server_path}", file=sys.stderr)
        sys.exit(1)

    if model_path is None or not os.path.isfile(model_path):
        print(f"missing model: {model_path}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "dflash" and (draft_path is None or not os.path.isfile(draft_path)):
        print(f"missing DFlash draft: {draft_path}", file=sys.stderr)
        sys.exit(1)

    # --- find device ---
    device_id, device_line = find_device(Path(server_path))
    print(f"selected device: {device_line}", file=sys.stderr)

    # --- preflight: verify binary commit matches worktree HEAD ---
    server_version = ""
    try:
        ver = subprocess.run(
            [str(server_path), "--version"],
            check=True,
            text=True,
            capture_output=True,
        )
        server_version = (ver.stdout.strip() or ver.stderr.strip())
        # Parse commit from version string (e.g., "llama-server version: 10581 (fe92d8a7f)")
        import re
        commit_match = re.search(r'\(([0-9a-f]{7,40})\)', server_version)
        if commit_match:
            binary_commit = commit_match.group(1)
            try:
                worktree_head = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT,
                    text=True
                ).strip()
                expected_head = expected_source_head(
                    args.external_source_head, worktree_head
                )
                if not expected_head.startswith(binary_commit):
                    print(
                        f"PREFLIGHT FAIL: binary commit {binary_commit} does not match "
                        f"expected HEAD {expected_head}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                print(f"preflight: binary commit {binary_commit} matches expected HEAD", file=sys.stderr)
            except Exception as exc:
                print(f"preflight: could not verify HEAD: {exc}", file=sys.stderr)
        else:
            print(f"preflight: could not parse commit from version: {server_version}", file=sys.stderr)
    except Exception as exc:
        print(f"preflight: could not get server version: {exc}", file=sys.stderr)

    # --- batch settings from environment ---
    batch_size = int(os.environ.get("VK_GAP_BATCH", "512"))
    ubatch_size = int(os.environ.get("VK_GAP_UBATCH", "128"))
    ctx_size = int(os.environ.get("VK_GAP_CTX_SIZE", "2048"))

    # --- build command ---
    all_args = build_server_command(
        server_path=server_path,
        model_path=model_path,
        draft_path=draft_path,
        port=port,
        n_parallel=np,
        device_id=device_id,
        mode=args.mode,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        ctx_size=ctx_size,
    )

    # --- output directory ---
    log_dir = ROOT / "bench" / "vulkan-gap" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{ts}-{args.mode}-{args.prompt}.log"

    output_path = Path(args.output) if args.output else None
    if output_path is None:
        output_path = ROOT / "bench" / "vulkan-gap" / "records" / f"{args.mode}-{args.prompt}-{ts}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Reject existing output to prevent stale-data contamination
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    # --- launch server ---
    log_fh = open(log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        all_args, stdout=log_fh, stderr=subprocess.STDOUT,
    )

    try:
        wait_ready(server_proc, port)
    except Exception as exc:
        print(f"server readiness failed: {exc}", file=sys.stderr)
        server_proc.terminate()
        server_proc.wait(timeout=5)
        sys.exit(1)

    # --- request helpers ---
    payload = {
        "prompt": None,  # filled below per prompt
        "n_predict": args.gen_tokens,
        "temperature": temp,
        "top_k": top_k,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 7,
        "stream": False,
        "cache_prompt": False,
        "return_tokens": True,
    }

    prompts = {
        "coding": (ROOT / "bench" / "vulkan-gap" / "prompts" / "coding.txt").read_text(encoding="utf-8").strip(),
        "math": (ROOT / "bench" / "vulkan-gap" / "prompts" / "math.txt").read_text(encoding="utf-8").strip(),
    }
    prompt_text = prompts[args.prompt]

    # --- warm-up request ---
    if not args.skip_warmup:
        warmup_payload = dict(payload, prompt=prompt_text, n_predict=args.warmup_tokens)
        try:
            http_json(f"http://127.0.0.1:{port}/completion", warmup_payload, timeout=600)
        except Exception as exc:
            print(f"warm-up request failed: {exc}", file=sys.stderr)
            server_proc.terminate()
            server_proc.wait(timeout=5)
            sys.exit(1)

    # --- measured requests ---
    payload["prompt"] = prompt_text
    payload["n_predict"] = args.gen_tokens

    # Create provenance record
    provenance = make_provenance_record(
        server_path=server_path,
        model_path=model_path,
        draft_path=draft_path,
        device_line=device_line,
        command=all_args,
        environment={k: v for k, v in os.environ.items() 
                    if k.startswith("GGML_DFLASH_") or k.startswith("VK_GAP_")},
        reference_label=args.reference_label,
        external_source_head=args.external_source_head,
    )

    records = []
    has_invalid = False
    try:
        with open_output_exclusive(output_path) as out_fh:
            for rep in range(1, args.repetitions + 1):
                # Optional server restart between repetitions
                if args.restart_between_reps and rep > 1:
                    print(f"rep {rep}: restarting server for clean state", file=sys.stderr)
                    server_proc.terminate()
                    server_proc.wait(timeout=5)
                    log_fh.close()
                    server_proc, log_fh = _restart_server(all_args, log_path, port)

                rep_payload = dict(payload)
                try:
                    response = http_json(
                        f"http://127.0.0.1:{port}/completion",
                        rep_payload,
                        timeout=600,
                    )
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    print(f"rep {rep}: request failed: {exc}", file=sys.stderr)
                    # Record the error
                    record = {
                        "valid": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "rep": rep,
                        "provenance": provenance,
                    }
                    out_fh.write(json.dumps(record) + "\n")
                    records.append(record)
                    has_invalid = True
                    continue
                except Exception as exc:
                    print(f"rep {rep}: unexpected error: {exc}", file=sys.stderr)
                    # Record the error
                    record = {
                        "valid": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "rep": rep,
                        "provenance": provenance,
                    }
                    out_fh.write(json.dumps(record) + "\n")
                    records.append(record)
                    has_invalid = True
                    continue

                measurement = extract_measurement(response, args.prompt, args.mode)
                
                # Always record, even if invalid
                record = {
                    "mode": args.mode,
                    "prompt": args.prompt,
                    "rep": rep,
                    "request": rep_payload,
                    "measurement": measurement,
                    "log": str(log_path),
                    "provenance": provenance,
                }
                records.append(record)
                out_fh.write(json.dumps(record) + "\n")
                
                if not measurement["valid"]:
                    has_invalid = True
                valid_str = "OK" if measurement["valid"] else f"INVALID ({', '.join(measurement['invalid_reasons'])})"
                print(f"rep {rep}: t/s={measurement['predicted_per_second']:.2f} "
                      f"toks={measurement['tokens_predicted']} "
                      f"finish={measurement['finish_reason']} {valid_str}")

        print(f"wrote {len(records)} records to {output_path}")
    finally:
        # Write manifest alongside JSONL (always, even on partial runs)
        try:
            _write_manifest(output_path, provenance, args, records)
        except Exception:
            pass
        # --- cleanup: always terminate server and close log ---
        try:
            server_proc.terminate()
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=5)
        except Exception:
            pass
        try:
            log_fh.close()
        except Exception:
            pass

    # Exit nonzero if we produced no records OR any measured row is invalid
    if not records or has_invalid:
        sys.exit(1)


def _write_manifest(output_path, provenance, args, records):
    """Write a manifest.json alongside the JSONL output."""
    manifest_path = output_path.with_suffix(".manifest.json")
    valid_count = sum(1 for r in records if r.get("measurement", {}).get("valid", False))
    invalid_count = len(records) - valid_count
    invalid_reasons = {}
    for r in records:
        for reason in r.get("measurement", {}).get("invalid_reasons", []):
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    manifest = {
        "output_path": str(output_path),
        "provenance": provenance,
        "args": {
            "mode": args.mode,
            "prompt": args.prompt,
            "repetitions": args.repetitions,
            "gen_tokens": args.gen_tokens,
        },
        "total_records": len(records),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "invalid_reasons": invalid_reasons,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)


def _restart_server(all_args, log_path, port):
    """Restart server and wait for readiness."""
    log_fh = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(all_args, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        wait_ready(proc, port)
    except Exception as exc:
        print(f"server readiness failed: {exc}", file=sys.stderr)
        proc.terminate()
        proc.wait(timeout=5)
        sys.exit(1)
    return proc, log_fh


if __name__ == "__main__":
    main()
