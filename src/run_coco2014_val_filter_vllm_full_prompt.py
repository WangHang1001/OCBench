#!/usr/bin/env python3
"""Run COCO2014 val image filtering with Qwen3-VL-235B and vLLM.

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
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "coco2014_val_qwen3vl235b_selection_full_prompt.json"
DEFAULT_CHECKPOINT_JSONL = DEFAULT_CHECKPOINT_DIR / "coco2014_val_qwen3vl235b_selection_full_prompt.jsonl"
DEFAULT_MODEL_NAME = "Qwen3-VL-235B-A22B-Thinking"

ALLOWED_DIFFICULTY_TYPES = [
    "large quantity",
    "scale variation",
    "similar-object confusion",
    "similar background",
    "clustered stacking",
    "occlusion or truncation",
]

DEFAULT_PROMPT = """You are an expert visual dataset annotator. Your task is to determine whether the given image is suitable for inclusion in a candidate image subset for a benchmark of challenging target-object counting for LVLMs/MLLMs.

This is an INITIAL FILTERING step before human review.
Your goal is to keep images that are likely useful for constructing genuinely challenging counting questions, while rejecting images that are clearly easy, clearly invalid, or not reliably countable.

Important overall principle:
A visually rich or cluttered image is NOT automatically a challenging counting image.
Select an image only when at least one target object category is:
1. a real physical object category;
2. clearly nameable;
3. present with at least two real instances;
4. reliably countable by humans with a stable exact ground-truth count;
5. not countable at a glance;
6. difficult due to at least one specific target-level difficulty type listed below.

This is a candidate-generation stage, not the final benchmark construction stage.
So keep images that are clearly or plausibly challenging and annotatable, but reject images that are obviously easy or whose exact count is not reliably annotatable.

========================
Step 1: Candidate target identification
========================

First identify one or more candidate target object categories that could reasonably be used in a counting question such as:
“How many [target objects] are in the image?”

Evaluate each candidate target category independently.
Do NOT transfer difficulty from one category to another.
Only categories that are themselves challenging should be retained.

========================
Step 2: Basic validity requirements
========================

A candidate target category is valid only if ALL of the following hold:

1. Real physical instance rule
Count only real physical instances in the scene.
Do NOT count reflections, mirror images, shadows, printed images, screen images, posters, paintings, or duplicated appearances as separate target instances.

2. Minimum count rule
The category must contain at least two real target instances.

3. Semantic clarity rule
The category must be semantically clear and visually consistent.
Reject categories whose membership is ambiguous in the image.

4. Annotatability rule
The exact count must be reasonably and reliably annotatable by humans.
If human annotators would likely disagree on the exact number, reject that target category.

Reject the target category if:
- instance boundaries are too unclear;
- the category is mixed with similar object types and cannot be stably defined;
- the count would depend on arbitrary judgment;
- instances are too blurry, too tiny, too fragmented, too hidden, or too merged to support a stable exact count.

Examples of often invalid targets when not clearly separable:
- distant trees in a background forest;
- blurry crowd-like background people;
- fragmented basil leaves or herb toppings;
- crumbs, grains, seeds, chopped food, grass, leaves, petals, or similar fragment-like objects;
- ambiguous container-like categories such as baskets/containers/trays/bowls/racks when category boundaries are unclear.

========================
Step 3: Easy-count rejection
========================

Reject a candidate target category if the count is easy.

At-a-glance rule:
If an average human can count the target objects immediately without careful searching, careful separation, or category judgment, then the target is NOT challenging.

Reject the target category if any of the following holds:
- only 2 or 3 target instances, and they are large, salient, clearly visible, and countable at a glance;
- the overall scene looks complex, but the target category itself is easy to count;
- the only apparent difficulty comes from general background clutter or rich scene content;
- the objects are regularly arranged and clearly separable;
- the objects are stacked or adjacent but still clearly separable;
- target and non-target objects are easy to distinguish.

If an image contains both difficult and easy countable categories, keep only the difficult categories.

========================
Step 4: Allowed difficulty types
========================

Use ONLY the following difficulty types:

1. large quantity
2. scale variation
3. similar-object confusion
4. similar background
5. clustered stacking
6. occlusion or truncation

A category is challenging only if at least one of these difficulty types directly affects counting that category.

------------------------
1. large quantity
------------------------
Use this when the target category contains many real, individually countable instances.

Guidelines:
- 2–3 instances: never large quantity
- 4–5 instances: usually not large quantity
- 6–9 instances: may be large quantity only if counting is not obvious and some additional difficulty exists
- 10 or more instances: generally can be large quantity, if still reliably annotatable

Do NOT use this type for:
- blurry crowds,
- indistinct background objects,
- fragmented objects,
- scenes where the exact count is not reliably annotatable.

------------------------
2. scale variation
------------------------
Use this when same-category target instances show substantial apparent-size variation, often due to perspective, depth, or distance.

This type is especially relevant when:
- some target instances are large/salient,
- other target instances are much smaller or less salient,
- the target instances are spatially distributed across different regions or depth levels,
- the smaller instances are easy to miss,
- exact counting requires careful whole-image search.

Do NOT use this type merely because:
- targets are small;
- targets are spatially separated;
- size differences are only mild;
- the count is still obvious at a glance.

Reject the target if the smallest instances are so tiny, blurry, or ambiguous that humans cannot reliably annotate the exact count.

------------------------
3. similar-object confusion
------------------------
Use this when visually similar NON-target objects create genuine ambiguity about whether they should be counted as target instances.

The confusion must be real, not superficial.

Do NOT use this type merely because:
- target and non-target objects belong to the same broad superclass (e.g. both are people, animals, vehicles, or food),
- non-target objects are clearly distinguishable by clothing, role, equipment, color, shape, position, or context.

Example:
A clearly dressed referee should usually NOT be treated as a confusing distractor for baseball players.

------------------------
4. similar background
------------------------
Use this when the target instances blend into their immediate surrounding background, supporting surface, or nearby region due to similar color, brightness, texture, or boundary appearance, making them easy to miss or hard to separate.

Do NOT use this type merely because:
- the whole image is grayscale, monochrome, low-saturation, or stylized;
- the target and background share a similar color but the target still has clear contours, shadows, or strong local contrast;
- the target forms a clear silhouette.

If the target remains clearly visible and countable at a glance, do NOT use this type.

------------------------
5. clustered stacking
------------------------
Use this when same-category target instances are densely packed, piled, stacked, touching, overlapping, or visually merged in a way that makes individual instances hard to separate and count.

The key is genuine instance-separation difficulty.

Do NOT use this type for:
- simple adjacency,
- side-by-side people,
- regular rows/columns/grids,
- clear stacking where each object still has visible edges, distinct layers, or obvious boundaries.

Reject the target if the stacking is so severe that humans cannot reliably determine the exact count.

------------------------
6. occlusion or truncation
------------------------
Use this when target instances are partially hidden by non-target objects, scene elements, or image boundaries, and that partial visibility makes target identification or exact counting genuinely difficult.

Use this type only when:
- enough visual evidence remains for humans to infer a stable exact count,
- but the partial visibility still makes counting meaningfully difficult.

Do NOT use this type if:
- the target is still clearly identifiable and countable at a glance,
- the occlusion/truncation is too mild to matter.

Reject the target if:
- so little visual evidence remains that humans cannot reliably determine the exact count.

Distinguish this from clustered stacking:
- use "occlusion or truncation" when the main difficulty is partial visibility caused by non-target objects, scene elements, or image boundaries;
- use "clustered stacking" when the main difficulty is separation among multiple same-category target instances.

Both labels may be used only if both independently contribute to the difficulty.

========================
Step 5: Key veto rules
========================

The following veto rules override all difficulty types:

1. easy-count veto
Reject if the target count is obvious at a glance.

2. reflection / duplicate-appearance veto
Reject if the apparent difficulty comes mainly from reflections, mirror images, shadows, screen images, posters, or other duplicated appearances rather than real additional physical instances.

3. unannotatable-target veto
Reject if humans would likely disagree on the exact count.

4. semantic-ambiguity veto
Reject if it is unclear which objects belong to the target category.

5. regular-layout veto
Reject if targets are regularly arranged and clearly separable.

6. clear-stacking veto
Reject if targets are stacked or overlapping but still clearly separable.

7. easy-distinction veto
Reject if target and non-target objects are easy to distinguish.

8. background-complexity veto
Reject if the overall scene is complex but the target category itself remains easy to count.

9. fragmented-object veto
Reject fragment-like or topping-like categories if they are not complete, clearly separated, and consistently countable.

========================
Step 6: Final selection rule
========================

Set "selected" to true only if the image contains at least one target category that:

1. satisfies the basic validity requirements;
2. has at least two real physical instances;
3. is not easy to count at a glance;
4. is not rejected by any veto rule;
5. has at least one valid difficulty type that directly affects counting that target category.

This is an initial candidate selection stage before human review.
So if a target category appears to be genuinely challenging and reliably annotatable, it is acceptable to keep it even if the case is not absolutely extreme.

However, when a case is clearly easy or clearly not reliably annotatable, reject it.

========================
Output format
========================

Output JSON only. Do not include markdown or extra explanation.

Use this exact structure:

{
  "selected": true/false,
  "final_selected_targets": ["..."],
  "difficulty_types_by_target": {
    "...": ["large quantity", "scale variation", "similar-object confusion", "similar background", "clustered stacking", "occlusion or truncation"]
  },
  "rejected_targets": ["..."],
  "brief_reason": "brief explanation"
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-after", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--record-errors", action="store_true", default=True, help="Record per-image errors and continue. This is the default.")
    parser.add_argument("--fail-on-error", action="store_true", help="Stop the run when a single image fails.")
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
            if args.fail_on_error:
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
                if args.fail_on_error:
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
