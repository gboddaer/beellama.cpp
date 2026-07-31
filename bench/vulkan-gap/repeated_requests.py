#!/usr/bin/env python3
"""Persistent-server and concurrent correctness driver.

Tests MTP/DFlash correctness across repeated and concurrent requests without
server restarts.
"""
import sys
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import concurrent.futures
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path):
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def http_json(url, payload, timeout=600):
    """POST JSON payload to url, return parsed JSON."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def wait_ready(port, timeout_s=180):
    """Wait for server to become healthy."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            http_json(f"http://127.0.0.1:{port}/health", {}, timeout=2)
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
        return [
            "--spec-type", "dflash",
            "--spec-draft-model", str(draft),
            "--spec-draft-n-max", "8",
        ]
    raise ValueError(f"Unknown mode: {mode}")


def validate_coding_response(content, finish_reason, tokens):
    """Validate a coding response."""
    reasons = []
    if finish_reason != "stop":
        reasons.append(f"finish_reason={finish_reason}, expected stop")
    if tokens < 10:
        reasons.append(f"too few tokens: {tokens}")
    if "def fibonacci" not in content:
        reasons.append("missing def fibonacci")
    if "Only output the code" in content and content.count("Only output") > 3:
        reasons.append("prompt echo detected")
    return reasons


def validate_math_response(content, finish_reason, tokens):
    """Validate a math response."""
    reasons = []
    if tokens < 5:
        reasons.append(f"too few tokens: {tokens}")
    if "2.4" not in content.lower():
        reasons.append("missing correct answer 2.4")
    if "def fibonacci" in content:
        reasons.append("contains coding code (cross-contamination)")
    return reasons


def run_single_request(port, prompt_type, prompt_text, n_tokens, mode, rep_id):
    """Run a single request and return results."""
    payload = {
        "prompt": prompt_text,
        "n_predict": n_tokens,
        "temperature": 0,
        "top_k": 20,
        "seed": 7,
        "stream": False,
        "return_tokens": True,
    }
    try:
        response = http_json(
            f"http://127.0.0.1:{port}/completion",
            payload,
            timeout=600,
        )
        timing = response.get("timings", {})
        # non-oaicompat format: flat content, tokens, stop
        content = response.get("content", "")
        stop_type = response.get("stop", False)
        finish_reason = "stop" if stop_type else "length"
        token_ids = [int(t) for t in response.get("tokens", [])] if response.get("tokens") else []
        tokens = response.get("tokens_predicted", 0)
        result = {
            "prompt_type": prompt_type,
            "rep_id": rep_id,
            "mode": mode,
            "valid": True,
            "reasons": [],
            "finish_reason": finish_reason,
            "tokens": tokens,
            "token_ids": token_ids,
            "tps": timing.get("predicted_per_second", 0),
            "draft_n": timing.get("draft_n", 0),
            "content": content,
        }
        
        # Validate
        if prompt_type == "coding":
            result["reasons"] = validate_coding_response(
                choice["text"], choice.get("finish_reason"), result["tokens"]
            )
            result["valid"] = len(result["reasons"]) == 0
        elif prompt_type == "math":
            result["reasons"] = validate_math_response(
                choice["text"], choice.get("finish_reason"), result["tokens"]
            )
            result["valid"] = len(result["reasons"]) == 0
        
        return result
    except Exception as exc:
        return {
            "prompt_type": prompt_type,
            "rep_id": rep_id,
            "mode": mode,
            "valid": False,
            "reasons": [f"Request failed: {exc}"],
            "error": str(exc),
            "finish_reason": None,
            "tokens": 0,
            "token_ids": [],
            "tps": 0,
            "draft_n": 0,
            "content": "",
        }


def run_persistent(args):
    """Run persistent server correctness tests."""
    # Get configuration
    server_path = Path(os.environ.get("VK_GAP_SERVER"))
    model_path = Path(os.environ.get("VK_GAP_MODEL"))
    draft_path = Path(os.environ.get("VK_GAP_DRAFT")) if os.environ.get("VK_GAP_DRAFT") else None
    port = int(os.environ.get("VK_GAP_PORT", "8099"))
    np = int(os.environ.get("VK_GAP_NP", "1"))

    # Find device
    device_id, device_line = find_device(server_path)

    # Batch settings from environment
    batch_size = int(os.environ.get("VK_GAP_BATCH", "512"))
    ubatch_size = int(os.environ.get("VK_GAP_UBATCH", "128"))
    ctx_size = int(os.environ.get("VK_GAP_CTX_SIZE", "2048"))

    # Build command
    common = [
        str(server_path), "-m", str(model_path),
        "--port", str(port), "-np", str(np),
        "--kv-unified", "-ngl", "all",
        "-b", str(batch_size), "-ub", str(ubatch_size),
        "--ctx-size", str(ctx_size),
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        "--cache-ram", "0", "--flash-attn", "on", "--device", device_id,
        "--reasoning", "off", "--no-mmap", "--no-host", "--host", "127.0.0.1",
    ]
    all_args = common + mode_args(args.mode, draft_path)

    # Log directory
    log_dir = ROOT / "bench" / "vulkan-gap" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{ts}-{args.mode}-persistent-{ts}.log"

    # Launch server
    print(f"Starting server for {args.mode} mode with -np {np}", file=sys.stderr)
    log_fh = open(log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        all_args, stdout=log_fh, stderr=subprocess.STDOUT,
    )

    try:
        wait_ready(port)
        
        # Get prompts
        prompts = {
            "coding": (ROOT / "bench" / "vulkan-gap" / "prompts" / "coding.txt").read_text().strip(),
            "math": (ROOT / "bench" / "vulkan-gap" / "prompts" / "math.txt").read_text().strip(),
        }
        
        # Warm-up
        print("Running warm-up request", file=sys.stderr)
        run_single_request(port, "coding", prompts["coding"], 32, args.mode, 0)
        
        # Measure
        prompt_type = args.prompt if hasattr(args, 'prompt') and args.prompt else "coding"
        prompt_text = prompts[prompt_type]
        n_tokens = getattr(args, 'gen_tokens', 512)
        
        results = []
        all_valid = True
        
        for rep in range(1, args.rounds + 1):
            print(f"Running persistent request {rep}/{args.rounds}", file=sys.stderr)
            result = run_single_request(
                port, prompt_type, prompt_text, n_tokens, args.mode, rep
            )
            results.append(result)
            if not result["valid"]:
                all_valid = False
                print(f"  FAIL: {result['reasons']}", file=sys.stderr)
            else:
                print(f"  OK: {result['tokens']} tokens, {result['tps']:.1f} t/s, "
                      f"draft_n={result.get('draft_n', 0)}", file=sys.stderr)
        
        return results, all_valid
        
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        log_fh.close()


def concurrent_request_specs(prompts, gen_tokens):
    """Return one coding and one math request spec for concurrent testing."""
    return [
        ("coding", prompts["coding"], gen_tokens),
        ("math", prompts["math"], gen_tokens),
    ]


def run_concurrent(args):
    """Run concurrent server correctness tests."""
    # Get configuration
    server_path = Path(os.environ.get("VK_GAP_SERVER"))
    model_path = Path(os.environ.get("VK_GAP_MODEL"))
    draft_path = Path(os.environ.get("VK_GAP_DRAFT")) if os.environ.get("VK_GAP_DRAFT") else None
    port = int(os.environ.get("VK_GAP_PORT", "8099"))
    np = int(os.environ.get("VK_GAP_NP", "2"))

    # Find device
    device_id, device_line = find_device(server_path)

    # Batch settings from environment
    batch_size = int(os.environ.get("VK_GAP_BATCH", "512"))
    ubatch_size = int(os.environ.get("VK_GAP_UBATCH", "128"))
    ctx_size = int(os.environ.get("VK_GAP_CTX_SIZE", "2048"))

    # Build command
    common = [
        str(server_path), "-m", str(model_path),
        "--port", str(port), "-np", str(np),
        "--kv-unified", "-ngl", "all",
        "-b", str(batch_size), "-ub", str(ubatch_size),
        "--ctx-size", str(ctx_size),
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        "--cache-ram", "0", "--flash-attn", "on", "--device", device_id,
        "--reasoning", "off", "--no-mmap", "--no-host", "--host", "127.0.0.1",
    ]
    all_args = common + mode_args(args.mode, draft_path)

    # Log directory
    log_dir = ROOT / "bench" / "vulkan-gap" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{ts}-{args.mode}-concurrent-{ts}.log"

    # Launch server
    print(f"Starting server for {args.mode} mode with -np {np}", file=sys.stderr)
    log_fh = open(log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        all_args, stdout=log_fh, stderr=subprocess.STDOUT,
    )

    try:
        wait_ready(port)
        
        # Get prompts
        prompts = {
            "coding": (ROOT / "bench" / "vulkan-gap" / "prompts" / "coding.txt").read_text().strip(),
            "math": (ROOT / "bench" / "vulkan-gap" / "prompts" / "math.txt").read_text().strip(),
        }
        
        all_results = []
        all_valid = True
        
        for round_num in range(1, args.rounds + 1):
            print(f"Running concurrent round {round_num}/{args.rounds}", file=sys.stderr)
            
            round_results = []
            
            # Send both requests concurrently
            specs = concurrent_request_specs(prompts, args.gen_tokens)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                for i, (pt, pt_text, n_tok) in enumerate(specs):
                    tag = "c" if pt == "coding" else "m"
                    futures[pt] = executor.submit(
                        run_single_request, port, pt, pt_text, n_tok,
                        args.mode, f"r{round_num}_{tag}"
                    )
                coding_result = futures["coding"].result(timeout=120)
                math_result = futures["math"].result(timeout=120)
                
                coding_result = coding_future.result(timeout=120)
                math_result = math_future.result(timeout=120)
                
                round_results.extend([coding_result, math_result])
                all_results.extend([coding_result, math_result])
                
                # Check for cross-contamination
                for result in round_results:
                    if result.get("valid"):
                        if result["prompt_type"] == "coding" and "2.4 hours" in result.get("content", ""):
                            result["valid"] = False
                            result["reasons"].append("math text in coding output")
                        elif result["prompt_type"] == "math" and "def fibonacci" in result.get("content", ""):
                            result["valid"] = False
                            result["reasons"].append("coding code in math output")
                
                if not all(r.get("valid", False) for r in round_results):
                    all_valid = False
                    for r in round_results:
                        if not r.get("valid"):
                            print(f"  {r['prompt_type']} FAIL: {r['reasons']}", file=sys.stderr)
                        else:
                            print(f"  {r['prompt_type']} OK", file=sys.stderr)
                else:
                    print(f"  All concurrent requests passed", file=sys.stderr)
        
        return all_results, all_valid
        
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        log_fh.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Persistent-server and concurrent correctness testing."
    )
    parser.add_argument("mode", choices=["base", "mtp", "dflash"])
    parser.add_argument("--np", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--prompt", choices=["coding", "math"], default="coding")
    parser.add_argument("--gen-tokens", type=int, default=512)
    parser.add_argument("--concurrent", action="store_true", help="Test concurrent requests")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file path")
    args = parser.parse_args()

    # Set environment variables for server configuration
    os.environ["VK_GAP_NP"] = str(args.np)

    if args.concurrent:
        results, all_valid = run_concurrent(args)
    else:
        results, all_valid = run_persistent(args)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = ROOT / "bench" / "vulkan-gap" / "records"
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        output_path = output_dir / f"correctness-{args.mode}-{'conc' if args.concurrent else 'pers'}-{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    # Record launched command for provenance
    provenance_cmd = (all_args if 'all_args' in dir() else [])

    with open(output_path, "x", encoding="utf-8") as fh:
        json.dump({
            "mode": args.mode,
            "np": args.np,
            "rounds": args.rounds,
            "concurrent": args.concurrent,
            "command": provenance_cmd,
            "results": results,
            "all_valid": all_valid,
            "valid_count": sum(1 for r in results if r.get("valid")),
            "total_count": len(results),
        }, fh, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print(f"Summary: {sum(1 for r in results if r.get('valid'))}/{len(results)} valid")
    
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
