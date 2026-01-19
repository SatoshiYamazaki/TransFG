# Quickstart (Phase 1)

## Environment
1) Create env (conda recommended):
```
conda env create -f environment.yml  # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
2) Capture env hash for metadata:
```
conda env export --from-history > env_export.yml
sha256sum env_export.yml > env_export.sha
```

## Tests (TDD first)
- Run smoke training + eval markers:
```
pytest -m "smoke or smoke_train or smoke_eval" -q
```

## Tiny Training (fp32, M2)
```
DATA_ROOT=${DATA_ROOT:-/path/to/data}
python train.py --name flower_mps \
  --dataset flower102 --data_root "$DATA_ROOT"/flower102 --model_type testing --img_size 64 \
  --num_steps 10 --eval_every 5 --train_batch_size 4 --eval_batch_size 2 \
  --output_dir output --prefer_mps \
  --tiny_train_subset flower102_tiny --tiny_infer_subset flower102_tiny
```
Artifacts: TensorBoard at `output/tb/flower_mps`; predictions at `output/flower_mps/fiftyone/predictions.jsonl`; env stamp at `output/flower_mps/env_stamp.json`.

## Eval-only (separate CLI)
```
DATA_ROOT=${DATA_ROOT:-/path/to/data}
python eval.py --name flower_eval --dataset flower102 --data_root "$DATA_ROOT"/flower102 --img_size 64 --eval_batch_size 2 \
  --checkpoint output/flower_mps/checkpoints/ckpt.bin \
  --output_dir output --prefer_mps --tiny_infer_subset flower102_tiny
```
Artifacts: TensorBoard at `output/tb/flower_eval`; predictions at `output/flower_eval/fiftyone/predictions.jsonl`; env stamp at `output/flower_eval/env_stamp.json`.

## Retention note
- Keep TensorBoard/FiftyOne artifacts ≥90 days. Store `RETENTION.txt` alongside run outputs (e.g., `output/<run>/RETENTION.txt`) stating TTL and storage location; cite this in PRs.

## Notes
- Device auto-selects MPS, falls back to CPU; no CUDA/AMP required. To avoid machine-specific paths, set `DATA_ROOT` (e.g., in a git-ignored `.special_env` file with `dataroot=/your/path`) before running commands.
- Checkpoints are not committed; keep under `output/<run>/checkpoints/` and record SHA256 for reference.
