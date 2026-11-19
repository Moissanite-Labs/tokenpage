#!/usr/bin/env python3
"""Export ABIOS benchmark traffic to data/streams.json.

Needs an abios checkout at ABIOS_COMMIT. The committed data/streams.json
already holds the output, so measure.py runs without this step.

    python extract_streams.py --abios /path/to/abios
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ABIOS_COMMIT = "6da3a64"
HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abios", required=True, help="path to abios checkout or export")
    ap.add_argument("--out", default=str(HERE / "data" / "streams.json"))
    ap.add_argument("--commit", help="record this commit when --abios is not a git checkout")
    args = ap.parse_args()
    root = Path(args.abios).resolve()
    pkg = root / "scripts" / "protean-abios"
    sys.path.insert(0, str(pkg))

    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        head = args.commit or "unknown (not a git checkout)"

    from decoder_context import setup_sizes
    from encoders import all_encoders, encode_transcript
    from english import render_english_lossless
    from lexicon import load_lexicon
    from post import post_ack_line
    from schemas.synbios.fixtures import lifecycle_11, lifecycle_40, lifecycle_200

    lex = load_lexicon()
    refs = all_encoders(lex)["refs"]
    tasks = {}
    for name, fn in (
        ("lifecycle_11", lifecycle_11),
        ("lifecycle_40", lifecycle_40),
        ("lifecycle_200", lifecycle_200),
    ):
        msgs = fn()
        abios_lines = encode_transcript(refs, msgs)
        sizes = setup_sizes(msgs, abios_lines, lex)
        setup = post_ack_line(sizes["minimal_text"]) + sizes["minimal_text"]
        tasks[name] = {
            "count": len(msgs),
            "abios_setup": setup,
            "abios_refs": abios_lines,
            # Same construction as abios bench.py run_bench_for_transcript.
            "json_compact": [
                json.dumps({"action": a, **p}, sort_keys=True, separators=(",", ":"))
                for a, p in msgs
            ],
            "english_lossless": [render_english_lossless(a, p) for a, p in msgs],
            "bench_o200k_post_minimal": sizes["post_minimal"],
        }
    blob = {
        "source": "https://github.com/Moissanite-Labs/abios",
        "abios_commit_expected": ABIOS_COMMIT,
        "abios_commit_seen": head,
        "encoder": "refs (format tag E)",
        "tasks": tasks,
    }
    text = json.dumps(blob, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out} sha256={hashlib.sha256(text.encode()).hexdigest()[:16]} abios={head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
