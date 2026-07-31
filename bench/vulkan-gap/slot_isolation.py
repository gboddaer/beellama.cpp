"""Slot isolation driver for Vulkan speculative decoding correctness tests.

Sends deterministic /completion requests to a controlled physical-slot schedule
and records raw tokens, first token, and provenance for each request.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Import helpers from run.py
_run_spec = importlib.util.spec_from_file_location("vk_gap", HERE / "run.py")
_run_mod = importlib.util.module_from_spec(_run_spec)
_run_spec.loader.exec_module(_run_mod)

build_server_command = _run_mod.build_server_command
extract_measurement = _run_mod.extract_measurement
find_device = _run_mod.find_device
wait_ready = _run_mod.wait_ready
open_output_exclusive = _run_mod.open_output_exclusive
http_json = _run_mod.http_json


def slot_schedule(n_slots, rounds):
    """Generate a round-robin slot schedule.

    Args:
        n_slots: number of physical slots (must be positive).
        rounds: number of rounds (must be positive).

    Returns:
        List of slot IDs in round-robin order.
        E.g., slot_schedule(2, 2) -> [0, 1, 0, 1]
              slot_schedule(4, 2) -> [0, 1, 2, 3, 0, 1, 2, 3]
    """
    if n_slots <= 0 or rounds <= 0:
        raise ValueError("n_slots and rounds must be positive")
    return [slot for _ in range(rounds) for slot in range(n_slots)]


def first_divergence(reference, candidate):
    """Find the first index where two sequences differ.

    Args:
        reference: reference token sequence.
        candidate: candidate token sequence to compare.

    Returns:
        Index of first divergence, or length of shorter sequence if all
        overlapping elements match but lengths differ, or None if identical.
    """
    for i, (lhs, rhs) in enumerate(zip(reference, candidate)):
        if lhs != rhs:
            return i
    return None if len(reference) == len(candidate) else min(len(reference), len(candidate))


def main():
    parser = argparse.ArgumentParser(
        description="Slot isolation driver for Vulkan speculative decoding.",
    )
    parser.add_argument("mode", choices=["base", "mtp", "dflash"])
    parser.add_argument(
        "--np", type=int, default=2,
        help="Number of parallel slots (default: 2).",
    )
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="Number of rounds (default: 2).",
    )
    parser.add_argument(
        "--prompt", choices=["coding", "math"], default="coding",
        help="Prompt type (default: coding).",
    )
    parser.add_argument(
        "--gen-tokens", type=int, default=512,
        help="Tokens to generate per request (default: 512).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Server port (default: from VK_GAP_PORT or 8099).",
    )
    args = parser.parse_args()

    # Configuration from environment
    server_path = os.environ.get("VK_GAP_SERVER", str(HERE.parents[2] / "build" / "bin" / "llama-server"))
    model_path = os.environ.get("VK_GAP_MODEL")
    draft_path = os.environ.get("VK_GAP_DRAFT")
    port = args.port if args.port else int(os.environ.get("VK_GAP_PORT", "8099"))
    temp = float(os.environ.get("VK_GAP_TEMP", "0"))
    top_k = int(os.environ.get("VK_GAP_TOP_K", "20"))
    batch_size = int(os.environ.get("VK_GAP_BATCH", "512"))
    ubatch_size = int(os.environ.get("VK_GAP_UBATCH", "128"))
    ctx_size = int(os.environ.get("VK_GAP_CTX_SIZE", "2048"))

    # Preflight checks
    if not os.path.isfile(server_path):
        print(f"missing server: {server_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path or not os.path.isfile(model_path):
        print(f"missing model: {model_path}", file=sys.stderr)
        sys.exit(1)
    if args.mode == "dflash" and (not draft_path or not os.path.isfile(draft_path)):
        print(f"missing DFlash draft: {draft_path}", file=sys.stderr)
        sys.exit(1)

    # Find device
    device_id, device_line = find_device(Path(server_path))
    print(f"selected device: {device_line}", file=sys.stderr)

    # Build command
    all_args = build_server_command(
        server_path=server_path,
        model_path=model_path,
        draft_path=draft_path,
        port=port,
        n_parallel=args.np,
        device_id=device_id,
        mode=args.mode,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        ctx_size=ctx_size,
    )

    # Output path
    output_path = Path(args.output) if args.output else None
    if output_path is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        output_path = HERE / "records" / f"slot-{args.mode}-{args.prompt}-np{args.np}-{ts}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    # Log path
    log_dir = HERE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{ts}-slot-{args.mode}-{args.prompt}-np{args.np}.log"

    # Launch server
    log_fh = open(log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        all_args, stdout=log_fh, stderr=subprocess.STDOUT,
    )

    try:
        wait_ready(server_proc, port)

        # Load prompt
        prompts = {
            "coding": (HERE / "prompts" / "coding.txt").read_text(encoding="utf-8").strip(),
            "math": (HERE / "prompts" / "math.txt").read_text(encoding="utf-8").strip(),
        }
        prompt_text = prompts[args.prompt]

        # Generate slot schedule
        schedule = slot_schedule(args.np, args.rounds)
        print(f"slot schedule: {schedule}", file=sys.stderr)

        # Send requests
        records = []
        for req_idx, slot_id in enumerate(schedule):
            payload = {
                "prompt": prompt_text,
                "n_predict": args.gen_tokens,
                "temperature": temp,
                "top_k": top_k,
                "top_p": 1.0,
                "min_p": 0.0,
                "seed": 7,
                "stream": False,
                "cache_prompt": False,
                "return_tokens": True,
                "id_slot": slot_id,
            }

            print(f"request {req_idx}: slot={slot_id}", file=sys.stderr)
            response = http_json(f"http://127.0.0.1:{port}/completion", payload, timeout=600)

            measurement = extract_measurement(response, args.prompt, args.mode)
            token_ids = measurement.get("token_ids", [])
            first_token = token_ids[0] if token_ids else None
            token_hash = hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()[:16]

            record = {
                "request_ordinal": req_idx,
                "id_slot": slot_id,
                "mode": args.mode,
                "prompt": args.prompt,
                "first_token_id": first_token,
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "token_hash": token_hash,
                "valid": measurement.get("valid", False),
                "invalid_reasons": measurement.get("invalid_reasons", []),
                "draft_n": measurement.get("draft_n", 0),
                "draft_n_accepted": measurement.get("draft_n_accepted", 0),
                "finish_reason": measurement.get("finish_reason", None),
                "tps": measurement.get("tps", 0),
                "provenance": {
                    "server_path": server_path,
                    "model_path": model_path,
                    "draft_path": draft_path,
                    "device": device_line,
                    "command": all_args,
                    "port": port,
                    "np": args.np,
                    "gen_tokens": args.gen_tokens,
                    "temp": temp,
                    "top_k": top_k,
                    "seed": 7,
                },
            }
            records.append(record)
            print(
                f"  -> slot={slot_id} first_token={first_token} "
                f"tokens={len(token_ids)} valid={measurement.get('valid', False)} "
                f"token_hash={token_hash}",
                file=sys.stderr,
            )

        # Write output
        with open_output_exclusive(output_path) as out_fh:
            for record in records:
                out_fh.write(json.dumps(record) + "\n")

        print(f"output written: {output_path}", file=sys.stderr)
        print(f"server log: {log_path}", file=sys.stderr)
    finally:
        server_proc.terminate()
        server_proc.wait(timeout=5)
        log_fh.close()


if __name__ == "__main__":
    main()
