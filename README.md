# OCBench

OCBench uses Qwen3-VL-235B-A22B-Thinking with vLLM to filter COCO2014 val images for challenging target-object counting samples.

## Directory

```text
OCBench/
  model/        # Qwen3-VL-235B-A22B-Thinking
  dataset/      # COCO2014 val images
  src/          # download, demo, and filtering scripts
  output/       # final JSON files
  checkpoint/   # resumable JSONL checkpoints
```

## 1. Create Environment

Create a fresh conda environment on the GPU server:

```bash
conda create -n OCBench python=3.10 -y
conda activate OCBench
```

Install project dependencies:

```bash
cd OCBench
pip install -r requirements.txt
```

## 2. Download Model

Download Qwen3-VL-235B-A22B-Thinking to `model/`:

```bash
cd OCBench
bash src/download_qwen3_vl_235b.sh
```

## 3. Download COCO2014 Val

Download COCO2014 val images to `dataset/`:

```bash
cd OCBench
bash src/download_coco2014_val.sh
```

## 4. Run Quick Demo

Test the full pipeline on a few images before the full run:

```bash
cd OCBench
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/demo_filter_few_images.py --limit 3 --tensor-parallel-size 8
```

Demo output:

```text
output/coco2014_val_qwen3vl235b_demo.json
checkpoint/coco2014_val_qwen3vl235b_demo.jsonl
```

## 5. Compare Prompts On First 210 Images

Run the shorter prompt on the first 210 COCO val images:

```bash
cd OCBench
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/run_coco2014_val_filter_vllm_first210.py \
  --tensor-parallel-size 8 \
  --workers 4 \
  --overwrite
```

Output:

```text
output/coco2014_val_qwen3vl235b_selection_first210.json
checkpoint/coco2014_val_qwen3vl235b_selection_first210.jsonl
```

Run the longer prompt on the first 210 COCO val images:

```bash
cd OCBench
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/run_coco2014_val_filter_vllm_full_prompt_first210.py \
  --tensor-parallel-size 8 \
  --workers 4 \
  --overwrite
```

Output:

```text
output/coco2014_val_qwen3vl235b_selection_full_prompt_first210.json
checkpoint/coco2014_val_qwen3vl235b_selection_full_prompt_first210.jsonl
```

Use these two JSON files for manual prompt-quality comparison before running the full dataset.

## 6. Run Full Filtering

Run the full COCO2014 val filtering task with the shorter prompt:

```bash
cd OCBench
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/run_coco2014_val_filter_vllm.py \
  --tensor-parallel-size 8 \
  --workers 4
```

Final output:

```text
output/coco2014_val_qwen3vl235b_selection.json
checkpoint/coco2014_val_qwen3vl235b_selection.jsonl
```

Run the full COCO2014 val filtering task with the longer prompt:

```bash
cd OCBench
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/run_coco2014_val_filter_vllm_full_prompt.py \
  --tensor-parallel-size 8 \
  --workers 4
```

Final output:

```text
output/coco2014_val_qwen3vl235b_selection_full_prompt.json
checkpoint/coco2014_val_qwen3vl235b_selection_full_prompt.jsonl
```

The run can be resumed with the same command; processed image ids in the checkpoint will be skipped.
Adjust `--workers` if the vLLM server becomes unstable or GPU utilization is too low.
Per-image API or JSON parsing failures are recorded as `selected=false`, `logprob=null`, and `brief_reason="ERROR: ..."`, then the run continues. Use `--fail-on-error` only when debugging.

## Output Format

Each record has this format:

```json
{
  "image_id": "COCO_val2014_000000000042.jpg",
  "selected": true,
  "logprob": -0.0123,
  "final_selected_targets": ["..."],
  "difficulty_types_by_target": {
    "...": ["large quantity", "scale variation"]
  },
  "rejected_targets": ["..."],
  "brief_reason": "..."
}
```
