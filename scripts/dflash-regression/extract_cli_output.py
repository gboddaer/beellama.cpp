#!/usr/bin/env python3
import re
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("usage: extract_cli_output.py <input> <output>", file=sys.stderr)
    sys.exit(2)

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(errors="replace").replace("\b", "")

# Remove llama banner and command menu.
marker = "[Start thinking]"
idx = text.find(marker)
if idx >= 0:
    generated = text[idx:]
else:
    # Fallback: keep text after the prompt truncation marker if present.
    prompt_marker = "... (truncated)"
    idx = text.find(prompt_marker)
    generated = text[idx + len(prompt_marker):] if idx >= 0 else text

# Remove perf and process lifecycle lines.
lines = []
for line in generated.splitlines():
    if re.search(r"\[ Prompt: .* Generation: .*\]", line):
        continue
    if line.strip() in {"Exiting...", "Loading model..."}:
        continue
    if line.startswith("build      :") or line.startswith("model      :"):
        continue
    lines.append(line.rstrip())

normalized = "\n".join(lines).strip()
dst.write_text(normalized + "\n")
