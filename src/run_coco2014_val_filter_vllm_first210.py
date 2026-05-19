#!/usr/bin/env python3
"""Run first-210 COCO2014 val image filtering with Qwen3-VL-235B and vLLM.

The script can start a local vLLM OpenAI-compatible server, send each COCO val
image to the model, parse the model's JSON answer, and write one normalized
record per image. Results are checkpointed as JSONL so long runs can resume.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model" / "Qwen3-VL-235B-A22B-Thinking"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "coco2014"
DEFAULT_IMAGES_DIR = DEFAULT_DATA_ROOT / "val2014"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "coco2014_val_qwen3vl235b_selection_first210.json"
DEFAULT_CHECKPOINT_JSONL = DEFAULT_CHECKPOINT_DIR / "coco2014_val_qwen3vl235b_selection_first210.jsonl"
DEFAULT_MODEL_NAME = "Qwen3-VL-235B-A22B-Thinking"
DEFAULT_LIMIT = 210

ALLOWED_DIFFICULTY_TYPES = [
    "large quantity",
    "scale variation",
    "similar-object confusion",
    "similar background",
    "clustered stacking",
    "occlusion or truncation",
]

DEFAULT_PROMPT = """You are an expert visual dataset annotator. Your task is to determine whether the given image is suitable for constructing a challenging target-object counting question for LVLMs/MLLMs.

Use high-precision filtering. When uncertain, reject the image.

A selected image must contain at least one target category that satisfies ALL requirements:

1. The target category is clearly nameable.
2. The image contains at least two real physical instances of the target category.
3. The exact count can be reliably annotated by humans with high agreement.
4. The target count is not obvious at a glance.
5. The counting difficulty is caused by at least one specific target-level difficulty type listed below.

Do not select an image merely because the overall scene looks complex. General background clutter, many non-target objects, indoor/outdoor complexity, reflections, or rich scene content are not sufficient unless they directly make the selected target category hard to count.

Count only real physical instances. Do not count reflections, shadows, mirror images, screen images, printed images, posters, paintings, or duplicated visual appearances as separate instances.

Allowed difficulty types:

1. large quantity:
Many real, individually countable target instances. As a guideline, 2-3 instances are never large quantity; 4-5 are usually not large quantity; 6-9 require additional difficulty; 10 or more can be large quantity if the count remains reliably annotatable.

2. scale variation:
Same-category target instances show substantial apparent-size variation, often due to perspective, depth, or distance. Large/salient instances appear together with much smaller, less salient, or distant instances, and the targets are spatially distributed so that smaller instances are easy to miss and careful whole-image search is required.

3. similar-object confusion:
Visually similar non-target objects create genuine ambiguity about whether they should be counted as target instances. Do not use this type merely because objects belong to the same broad superclass, such as people, animals, vehicles, furniture, or food.

4. similar background:
Target instances blend into their immediate surrounding background, surface, container, or nearby objects due to similar color, brightness, texture, or boundary appearance, making them hard to detect or separate. Do not use this type if the target still has clear contours, shadows, silhouettes, or strong local contrast.

5. clustered stacking:
Same-category target instances are densely packed, piled, stacked, touching, overlapping, or visually merged, making individual instances hard to separate and count. Do not use this type for simple adjacency, regular rows/grids, or clear stacking where each instance remains separable.

6. occlusion or truncation:
Target instances are partially hidden by non-target objects, scene elements, or image boundaries, and the remaining visual evidence makes identification or exact enumeration genuinely difficult. Do not use this type if the target remains clearly identifiable and countable at a glance.

Reject the candidate target if any veto applies:

- easy-count veto: the count is obvious at a glance.
- small-count salient-object veto: there are only 2 or 3 target instances and they are large, salient, clearly visible, and easy to count.
- unannotatable target veto: humans would likely disagree on the exact count.
- semantic ambiguity veto: it is unclear which objects belong to the target category.
- reflection/duplicate veto: the apparent count comes from reflections, shadows, mirrors, screens, posters, or duplicated appearances.
- regular-layout veto: targets are regularly arranged and clearly separable.
- clear-stacking veto: objects are stacked or overlapping but each instance is still clearly separable.
- easy-distinction veto: target and non-target objects are easy to distinguish by clear visual cues.
- background-complexity veto: the background is complex but the target category itself remains easy to count.
- fragmented-object veto: targets such as leaves, herbs, grass, toppings, crumbs, grains, seeds, chopped food, or fragments are irregular, boundary-ambiguous, or not consistently countable.

Evaluate each candidate target category independently. Do not transfer difficulty from one target category to another. Include only categories that are valid challenging counting targets.

Output JSON only:

{
  "selected": true/false,
  "final_selected_targets": ["..."],
  "difficulty_types_by_target": {
    "...": ["large quantity", "scale variation", "similar-object confusion", "similar background", "clustered stacking", "occlusion or truncation"]
  },
  "rejected_targets": ["..."],
  "brief_reason": "..."
}"""


class AnnotationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter COCO2014 val images with Qwen3-VL via vLLM.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--checkpoint-jsonl", type=Path, default=DEFAULT_CHECKPOINT_JSONL)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-logprobs", action="store_true", help="Disable selected true/false token logprob collection.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--start-after", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--record-errors", action="store_true")
    parser.add_argument("--export-only", action="store_true")

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.getenv("LOCAL_VLM_API_KEY", "EMPTY"))
    parser.add_argument("--no-start-vllm", action="store_true", help="Use an already running vLLM server.")
    parser.add_argument("--keep-vllm", action="store_true", help="Do not terminate vLLM after this script exits.")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-log", type=Path, default=PROJECT_ROOT / "vllm_qwen3vl235b.log")
    parser.add_argument("--vllm-extra-arg", action="append", default=[], help="Extra argument passed to vLLM.")
    return parser.parse_args()


def load_prompt(prompt_file: Optional[Path]) -> str:
    return prompt_file.read_text(encoding="utf-8") if prompt_file else DEFAULT_PROMPT


def base_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}/v1"


def chat_url(args: argparse.Namespace) -> str:
    return base_url(args).rstrip("/") + "/chat/completions"


def server_ready(args: argparse.Namespace) -> bool:
    try:
        response = requests.get(base_url(args).rstrip("/") + "/models", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def start_vllm(args: argparse.Namespace) -> Optional[subprocess.Popen]:
    if server_ready(args):
        print(f"vLLM server already ready at {base_url(args)}")
        return None
    if args.no_start_vllm:
        raise RuntimeError(f"vLLM server is not ready at {base_url(args)}")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {args.model_dir}")

    args.vllm_log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.vllm_log.open("a", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model_dir),
        "--served-model-name",
        args.model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        "image=1",
    ]
    cmd.extend(args.vllm_extra_arg)
    print("Starting vLLM:")
    print(" ".join(cmd))
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    for second in range(1, 1801):
        if server_ready(args):
            print(f"vLLM server ready after {second}s at {base_url(args)}")
            return process
        if process.poll() is not None:
            log_file.close()
            tail = tail_text(args.vllm_log, 120)
            raise RuntimeError(f"vLLM exited with code {process.returncode}\n{tail}")
        if second % 30 == 0:
            print(f"Waiting for vLLM... {second}s")
        time.sleep(1)

    raise TimeoutError(f"Timed out waiting for vLLM at {base_url(args)}")


def stop_vllm(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def tail_text(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def list_images(images_dir: Path, start_after: Optional[str] = None) -> List[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    images = sorted(images_dir.glob("*.jpg"))
    return [p for p in images if not start_after or p.name > start_after]


def load_done_records(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: ignored invalid JSONL line {line_no}: {jsonl_path}", file=sys.stderr)
                continue
            image_id = record.get("image_id")
            if isinstance(image_id, str):
                done[image_id] = record
    return done


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(args: argparse.Namespace, prompt: str, image_path: Path) -> Dict[str, Any]:
    payload = {
        "model": args.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if not args.no_logprobs:
        payload["logprobs"] = True
    return payload


def request_headers(args: argparse.Namespace) -> Dict[str, str]:
    return {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}


def extract_message_text(response_json: Dict[str, Any]) -> str:
    content = response_json["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content)


def parse_model_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)


def decode_logprob_token(token_data: Dict[str, Any], prefer_bytes: bool) -> str:
    if prefer_bytes:
        raw_bytes = token_data.get("bytes")
        if isinstance(raw_bytes, list):
            try:
                return bytes(int(x) for x in raw_bytes).decode("utf-8", errors="replace")
            except (TypeError, ValueError):
                pass
    token = token_data.get("token", "")
    return token if isinstance(token, str) else str(token)


def selected_value_logprob(response_json: Dict[str, Any]) -> Optional[float]:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None

    selected_re = re.compile(r'"selected"\s*:\s*(true|false)\b', flags=re.IGNORECASE)
    for prefer_bytes in (True, False):
        spans = []
        parts = []
        cursor = 0
        for item in content:
            if not isinstance(item, dict):
                continue
            token_text = decode_logprob_token(item, prefer_bytes)
            start = cursor
            cursor += len(token_text)
            spans.append((start, cursor, item))
            parts.append(token_text)

        text = "".join(parts)
        match = selected_re.search(text)
        if not match:
            continue

        value_start, value_end = match.span(1)
        token_logprobs = []
        for start, end, item in spans:
            if start < value_end and end > value_start:
                token_logprob = item.get("logprob")
                if isinstance(token_logprob, (int, float)):
                    token_logprobs.append(float(token_logprob))
        if token_logprobs:
            return sum(token_logprobs)
    return None


def normalize_record(image_id: str, model_data: Dict[str, Any], logprob: Optional[float] = None) -> Dict[str, Any]:
    selected = normalize_bool(model_data.get("selected", False))

    final_selected_targets = model_data.get("final_selected_targets", [])
    if isinstance(final_selected_targets, str):
        final_selected_targets = [final_selected_targets]
    if not isinstance(final_selected_targets, list):
        final_selected_targets = []
    final_selected_targets = [str(x).strip() for x in final_selected_targets if str(x).strip()]

    raw_difficulty_map = model_data.get("difficulty_types_by_target", {})
    if not isinstance(raw_difficulty_map, dict):
        raw_difficulty_map = {}

    allowed = {x.lower(): x for x in ALLOWED_DIFFICULTY_TYPES}
    difficulty_types_by_target: Dict[str, List[str]] = {}
    for target, raw_types in raw_difficulty_map.items():
        target_name = str(target).strip()
        if not target_name:
            continue
        if isinstance(raw_types, str):
            raw_types = [raw_types]
        if not isinstance(raw_types, list):
            raw_types = []
        normalized_types = []
        for item in raw_types:
            key = str(item).strip().lower()
            if key in allowed and allowed[key] not in normalized_types:
                normalized_types.append(allowed[key])
        if normalized_types:
            difficulty_types_by_target[target_name] = normalized_types

    if selected and not final_selected_targets:
        final_selected_targets = list(difficulty_types_by_target.keys())

    if selected and final_selected_targets:
        final_selected_targets = [
            target
            for target in final_selected_targets
            if target in difficulty_types_by_target
        ]
        difficulty_types_by_target = {
            target: difficulty_types_by_target[target]
            for target in final_selected_targets
        }

    brief_reason = model_data.get("brief_reason", "")
    if not isinstance(brief_reason, str):
        brief_reason = json.dumps(brief_reason, ensure_ascii=False)
    brief_reason = " ".join(brief_reason.split())[:1000]

    rejected_targets = model_data.get("rejected_targets", [])
    if isinstance(rejected_targets, str):
        rejected_targets = [rejected_targets]
    if not isinstance(rejected_targets, list):
        rejected_targets = []
    rejected_targets = [str(x).strip() for x in rejected_targets if str(x).strip()]

    if not selected and final_selected_targets:
        final_selected_targets = []
        difficulty_types_by_target = {}

    return {
        "image_id": image_id,
        "selected": selected,
        "logprob": logprob,
        "final_selected_targets": final_selected_targets,
        "difficulty_types_by_target": difficulty_types_by_target,
        "rejected_targets": rejected_targets,
        "brief_reason": brief_reason,
    }


def error_record(image_id: str, message: str) -> Dict[str, Any]:
    return {
        "image_id": image_id,
        "selected": False,
        "logprob": None,
        "final_selected_targets": [],
        "difficulty_types_by_target": {},
        "rejected_targets": [],
        "brief_reason": f"ERROR: {message}"[:1000],
    }


def retryable(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def sleep_before_retry(base_delay: float, attempt: int, response: Optional[requests.Response]) -> None:
    retry_after = None
    if response is not None and response.headers.get("retry-after"):
        try:
            retry_after = float(response.headers["retry-after"])
        except ValueError:
            retry_after = None
    delay = retry_after if retry_after is not None else base_delay * (2**attempt)
    time.sleep(delay + random.uniform(0, min(1.0, delay * 0.1)))


def annotate_image(image_path: Path, prompt: str, args: argparse.Namespace, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    client = session or requests.Session()
    payload = build_payload(args, prompt, image_path)
    last_error = ""
    for attempt in range(args.max_retries + 1):
        try:
            response = client.post(chat_url(args), headers=request_headers(args), json=payload, timeout=args.timeout)
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
                if attempt < args.max_retries and retryable(response.status_code):
                    sleep_before_retry(args.retry_base_delay, attempt, response)
                    continue
                raise AnnotationError(last_error)
            response_json = response.json()
            model_data = parse_model_json(extract_message_text(response_json))
            return normalize_record(image_path.name, model_data, selected_value_logprob(response_json))
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, AnnotationError) as exc:
            last_error = str(exc)
            if attempt >= args.max_retries:
                raise AnnotationError(last_error) from exc
            sleep_before_retry(args.retry_base_delay, attempt, None)
    raise AnnotationError(last_error or "unknown error")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_output_json(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def export_checkpoint(checkpoint_jsonl: Path, output_json: Path) -> int:
    records = list(load_done_records(checkpoint_jsonl).values())
    records.sort(key=lambda x: x.get("image_id", ""))
    write_output_json(output_json, records)
    return len(records)


def pending_images(all_images: Iterable[Path], done: Dict[str, Dict[str, Any]], limit: Optional[int]) -> List[Path]:
    images = [p for p in all_images if p.name not in done]
    return images[:limit] if limit is not None else images


def run_serial(images: Sequence[Path], prompt: str, args: argparse.Namespace) -> None:
    session = requests.Session()
    for idx, image_path in enumerate(images, start=1):
        try:
            record = annotate_image(image_path, prompt, args, session)
        except AnnotationError as exc:
            if not args.record_errors:
                raise
            record = error_record(image_path.name, str(exc))
        append_jsonl(args.checkpoint_jsonl, record)
        print(f"[{idx}/{len(images)}] {record['image_id']} selected={record['selected']}", flush=True)


def run_parallel(images: Sequence[Path], prompt: str, args: argparse.Namespace) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(annotate_image, image_path, prompt, args, None): image_path for image_path in images}
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            image_path = futures[future]
            try:
                record = future.result()
            except AnnotationError as exc:
                if not args.record_errors:
                    raise
                record = error_record(image_path.name, str(exc))
            append_jsonl(args.checkpoint_jsonl, record)
            print(f"[{idx}/{len(images)}] {record['image_id']} selected={record['selected']}", flush=True)


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.overwrite:
        args.checkpoint_jsonl.unlink(missing_ok=True)
        args.output_json.unlink(missing_ok=True)
    if args.export_only:
        print(f"exported {export_checkpoint(args.checkpoint_jsonl, args.output_json)} records to {args.output_json}")
        return 0

    process = start_vllm(args)
    try:
        prompt = load_prompt(args.prompt_file)
        done = load_done_records(args.checkpoint_jsonl)
        all_images = list_images(args.images_dir, args.start_after)
        images = pending_images(all_images, done, args.limit)
        print(f"images_dir: {args.images_dir}")
        print(f"model_dir: {args.model_dir}")
        print(f"base_url: {base_url(args)}")
        print(f"already_done: {len(done)}")
        print(f"pending_this_run: {len(images)}")
        print(f"checkpoint_jsonl: {args.checkpoint_jsonl}")
        print(f"output_json: {args.output_json}")
        if images:
            run_serial(images, prompt, args) if args.workers == 1 else run_parallel(images, prompt, args)
        print(f"done. exported {export_checkpoint(args.checkpoint_jsonl, args.output_json)} records to {args.output_json}")
        return 0
    finally:
        if process is not None and not args.keep_vllm:
            stop_vllm(process)


if __name__ == "__main__":
    raise SystemExit(main())
