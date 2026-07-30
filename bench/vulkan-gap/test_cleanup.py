#!/usr/bin/env python3
"""Test process cleanup and error handling."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_missing_model():
    """Test that missing model causes clean exit."""
    env = os.environ.copy()
    env["VK_GAP_MODEL"] = "/nonexistent/model.gguf"
    env["VK_GAP_SERVER"] = str(ROOT / "build-vulkan" / "bin" / "llama-server")
    
    proc = subprocess.Popen(
        [str(ROOT / "bench" / "vulkan-gap" / "run.py"), "base", "coding"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    stdout, stderr = proc.communicate(timeout=10)
    
    assert proc.returncode != 0, "Should exit nonzero when model missing"
    assert b"missing model" in stderr, f"Expected 'missing model' in stderr, got: {stderr.decode()}"
    print("✅ Missing model test passed")


def test_server_lifecycle():
    """Test that server process is cleaned up."""
    # First, kill any existing server on port 8099
    subprocess.run(
        ["pkill", "-f", "llama-server.*8099"],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    
    # Start a server with a short timeout
    env = os.environ.copy()
    env["VK_GAP_SERVER"] = str(ROOT / "build-vulkan" / "bin" / "llama-server")
    env["VK_GAP_MODEL"] = "/crypt/models/Qwen3.6-27B-Q4_K_M.gguf"
    env["VK_GAP_PORT"] = "8099"
    
    proc = subprocess.Popen(
        [str(ROOT / "bench" / "vulkan-gap" / "run.py"), "base", "coding", "--skip-warmup"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server to start
    time.sleep(3)
    
    # Check if server is running
    server_running = proc.poll() is None
    print(f"Server running: {server_running}")
    
    # Terminate
    proc.terminate()
    try:
        proc.wait(timeout=5)
        print("✅ Server terminated cleanly")
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        print("✅ Server killed after timeout")
    
    # Check no server processes remain
    result = subprocess.run(
        ["pgrep", "-f", "llama-server.*8099"],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(f"❌ Server processes still running: {result.stdout}")
        sys.exit(1)
    else:
        print("✅ No server processes remaining")


if __name__ == "__main__":
    test_missing_model()
    test_server_lifecycle()
    print("\nAll cleanup tests passed! ✅")
