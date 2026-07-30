#!/usr/bin/env python3
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


def extract_measurement(response):
    """Extract measurement from either chat or completions response format."""
    timings = response.get("timings") or {}
    draft_n = int(timings.get("draft_n") or 0)
    accepted = int(timings.get("draft_n_accepted") or 0)

    # Extract content and tokens from either format
    content = ""
    tokens_predicted = 0

    if "choices" in response:
        # Completions or chat format
        choices = response.get("choices", [])
        if choices:
            choice = choices[0]
            content = choice.get("text", "") or choice.get("message", {}).get("content", "")
            tokens_predicted = response.get("usage", {}).get("completion_tokens", 0) or choice.get("tokens_predicted", 0)
    elif "content" in response:
        # Flat format
        content = response.get("content", "")
        tokens_predicted = int(response.get("tokens_predicted") or 0)

    return {
        "tokens_predicted": tokens_predicted,
        "predicted_per_second": float(timings.get("predicted_per_second") or 0.0),
        "draft_accept_pct": (100.0 * accepted / draft_n) if draft_n else None,
        "content_sha256": hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest(),
        "content": content,
    }


def summary_stats(values):
    med = statistics.median(values)
    deviations = [abs(value - med) for value in values]
    return {
        "median": med,
        "min": min(values),
        "max": max(values),
        "mad": statistics.median(deviations),
    }


def sha256_file(path):
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
    text = subprocess.run(
        [str(server), "--list-devices"], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout
    for line in text.splitlines():
        if "NAVI31" in line or "GFX1151" in line:
            return line.split(":", 1)[0].strip(), line.strip()
    raise RuntimeError("no Vulkan NAVI31/GFX1151 device found")


def mode_args(mode, draft):
    if mode == "base":
        return ["--spec-type", "none"]
    if mode == "mtp":
        return ["--spec-type", "draft-mtp", "--spec-draft-n-max", "8"]
    if mode == "dflash":
        return [
            "--spec-type", "dflash",
            "--spec-draft-model", str(draft),
            "--spec-draft-n-max", "8",
            "--spec-branch-budget", "0",
            "--spec-dflash-cross-ctx", "512",
        ]
    raise ValueError(mode)


def main():
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
    args = parser.parse_args()

    # --- configuration from environment ---
    server_path = os.environ.get("VK_GAP_SERVER", str(ROOT / "build" / "bin" / "llama-server"))
    model_path = os.environ.get("VK_GAP_MODEL")
    draft_path = os.environ.get("VK_GAP_DRAFT")
    port = int(os.environ.get("VK_GAP_PORT", "8099"))
    temp = float(os.environ.get("VK_GAP_TEMP", "0"))
    top_k = int(os.environ.get("VK_GAP_TOP_K", "20"))

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

    # --- build command ---
    common = [
        str(server_path), "-m", str(model_path),
        "--port", str(port), "-np", os.environ.get("VK_GAP_NP", "1"),
        "--kv-unified", "-ngl", "all", "-b", "2048", "-ub", "512",
        "--ctx-size", "8192", "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--cache-ram", "0",
        "--flash-attn", "on",
        "--device", device_id, "--jinja", "--reasoning", "off",
        "--no-mmap", "--no-host", "--host", "127.0.0.1",
    ]
    all_args = common + mode_args(args.mode, Path(draft_path) if draft_path else None)

    # --- output directory ---
    log_dir = ROOT / "bench" / "vulkan-gap" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{ts}-{args.mode}-{args.prompt}.log"

    output_path = Path(args.output) if args.output else None
    if output_path is None:
        output_path = ROOT / "bench" / "vulkan-gap" / "records" / f"{args.mode}-{args.prompt}-{ts}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    }

    prompts = {
        "coding": (ROOT / "bench" / "vulkan-gap" / "prompts" / "coding.txt").read_text(encoding="utf-8").strip(),
        "math": (ROOT / "bench" / "vulkan-gap" / "prompts" / "math.txt").read_text(encoding="utf-8").strip(),
    }
    prompt_text = prompts[args.prompt]

    # --- warm-up request ---
    # MTP and DFlash have a pre-existing upstream bug where a warm-up request
    # corrupts the next request's speculative state (stale embeddings / ring
    # buffer), producing garbage drafts. Skip warm-up entirely for spec modes
    # and rely on server restarts between reps for clean state.
    if args.mode == "base":
        warmup_payload = dict(payload, prompt=prompt_text, n_predict=args.warmup_tokens)
        try:
            http_json(f"http://127.0.0.1:{port}/v1/completions", warmup_payload, timeout=600)
        except Exception as exc:
            print(f"warm-up request failed: {exc}", file=sys.stderr)
            server_proc.terminate()
            server_proc.wait(timeout=5)
            sys.exit(1)
    else:
        print(f"skipping warm-up for {args.mode} (speculative state corruption workaround)", file=sys.stderr)

    # --- measured requests ---
    payload["prompt"] = prompt_text
    payload["n_predict"] = args.gen_tokens

    # MTP and DFlash both have a pre-existing upstream bug where speculative
    # state corrupts across requests when slots are reused. Restart server
    # between reps to guarantee clean state.
    restart_between_reps = args.mode in ("dflash", "mtp")

    def start_server():
        """Start (or restart) the server and wait for readiness."""
        if server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait(timeout=5)
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

    records = []
    with open(output_path, "a", encoding="utf-8") as out_fh:
        for rep in range(1, args.repetitions + 1):
            if restart_between_reps and rep > 1:
                print(f"rep {rep}: restarting server for clean state", file=sys.stderr)
                server_proc.terminate()
                server_proc.wait(timeout=5)
                server_proc, log_fh = start_server()
            rep_payload = dict(payload)
            try:
                response = http_json(
                    f"http://127.0.0.1:{port}/v1/completions",
                    rep_payload,
                    timeout=600,
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"rep {rep}: request failed: {exc}", file=sys.stderr)
                continue
            except Exception as exc:
                print(f"rep {rep}: unexpected error: {exc}", file=sys.stderr)
                continue

            measurement = extract_measurement(response)
            if measurement["tokens_predicted"] <= 0 or measurement["predicted_per_second"] <= 0:
                print(f"rep {rep}: invalid measurement (tp={measurement['tokens_predicted']}, tps={measurement['predicted_per_second']})", file=sys.stderr)
                continue

            record = {
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "server": str(server_path),
                "server_sha256": sha256_file(server_path),
                "device": device_line,
                "model": str(model_path),
                "draft": str(draft_path) if args.mode == "dflash" else None,
                "mode": args.mode,
                "prompt": args.prompt,
                "repetition": rep,
                "command": [str(a) for a in all_args],
                "request": rep_payload,
                "measurement": measurement,
                "log": str(log_path),
                "environment": {
                    key: value for key, value in os.environ.items()
                    if key.startswith("GGML_DFLASH_") or key.startswith("VK_GAP_")
                },
            }
            records.append(record)
            out_fh.write(json.dumps(record) + "\n")
            print(f"rep {rep}: t/s={measurement['predicted_per_second']:.2f} "
                  f"toks={measurement['tokens_predicted']} "
                  f"hash={measurement['content_sha256'][:8]}")

    print(f"wrote {len(records)} records to {output_path}")

    # --- cleanup ---
    server_proc.terminate()
    try:
        server_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        server_proc.wait(timeout=5)
    log_fh.close()

    # Exit nonzero if we produced no records
    if not records:
        sys.exit(1)


if __name__ == "__main__":
    main()
