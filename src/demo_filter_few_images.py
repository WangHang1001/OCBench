#!/usr/bin/env python3
"""Small demo for the OCBench COCO2014 val filtering pipeline.

It starts or reuses the vLLM server, runs the main filter on a few images, and
writes demo JSON files under output/ and checkpoint/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "src" / "run_coco2014_val_filter_vllm.py"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small OCBench filtering demo.")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--no-start-vllm", action="store_true")
    parser.add_argument("--keep-vllm", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_json = OUTPUT_DIR / "coco2014_val_qwen3vl235b_demo.json"
    checkpoint_jsonl = CHECKPOINT_DIR / "coco2014_val_qwen3vl235b_demo.jsonl"

    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--limit",
        str(args.limit),
        "--overwrite",
        "--output-json",
        str(output_json),
        "--checkpoint-jsonl",
        str(checkpoint_jsonl),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
    ]
    if args.no_start_vllm:
        cmd.append("--no-start-vllm")
    if args.keep_vllm:
        cmd.append("--keep-vllm")

    print("Running demo command:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        return completed.returncode

    print(f"Demo output: {output_json}")
    if output_json.exists():
        print(output_json.read_text(encoding="utf-8")[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
