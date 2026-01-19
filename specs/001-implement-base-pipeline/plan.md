# Implementation Plan: Mac M2 Base Pipeline Implementation

**Branch**: `001-implement-base-pipeline` | **Date**: 2026-01-19 | **Spec**: [specs/001-implement-base-pipeline/spec.md](specs/001-implement-base-pipeline/spec.md)
**Input**: Feature specification from `/specs/001-implement-base-pipeline/spec.md`

**Note**: Filled via `/speckit.plan` with TDD-first workflow on MacBook Air M2.

## Summary

Implement a Mac M2-friendly TransFG baseline with fp32-only training and a separate eval-only CLI. Provide pytest smoke tests on the flower102 dataset (tiny slices ≤2 batches/split) with data root provided via environment variable (e.g., `DATA_ROOT` in a git-ignored `.special_env` file), TensorBoard + FiftyOne artifacts, env_stamp metadata, and reproducible env artifacts. Outputs use the normal schema `output/<run>/tb`, `output/<run>/fiftyone/predictions.jsonl`, `output/<run>/labelmap.json`, `output/<run>/checkpoints/ckpt.bin`, and `output/<run>/env_stamp.json`. Development is TDD-first: tests/markers are authored or updated before code changes and must stay green for every slice.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.10+ (arm64)  
**Primary Dependencies**: PyTorch 2.x, torchvision 0.15+, ml_collections, tensorboard, pytest, optional fiftyone  
**Storage**: Local filesystem artifacts under `output/<run>/...`  
**Testing**: pytest (markers: `smoke_train`, `smoke_eval`, aggregate `smoke`), commands recorded in quickstart; TDD enforced (tests written/updated before implementation)  
**Target Platform**: macOS (MacBook Air M2, arm64, CPU/MPS); CPU fallback mandatory  
**Project Type**: Single CLI/training project  
**Performance Goals**: Smoke train ≤3 minutes, eval-only tiny slice ≤5 minutes on M2; full training not in scope  
**Constraints**: No CUDA assumptions; fp32 only (AMP removed); batch sizes small for M2 memory; deterministic seeding; retention evidence stored as `output/<run>/RETENTION.txt` and cited in PRs  
**Scale/Scope**: Flower102 tiny subset (≤2 batches) for smoke; other real datasets optional via user-supplied paths

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Training command (flower102 tiny smoke): `DATA_ROOT=${DATA_ROOT:-/path/to/data} python train.py --name flower_mps --dataset flower102 --data_root "$DATA_ROOT"/flower102 --model_type testing --img_size 64 --num_steps 10 --eval_every 5 --train_batch_size 4 --eval_batch_size 2 --tiny_train_subset flower102_tiny --tiny_infer_subset flower102_tiny --output_dir output --prefer_mps` (fp32). Artifacts under `output/flower_mps/...`; env stamp at `output/flower_mps/env_stamp.json` with `fp16=false` recorded.
- Eval-only command (flower102 tiny): `DATA_ROOT=${DATA_ROOT:-/path/to/data} python eval.py --name flower_eval --dataset flower102 --data_root "$DATA_ROOT"/flower102 --model_type testing --img_size 64 --eval_batch_size 2 --checkpoint output/flower_mps/checkpoints/ckpt.bin --tiny_infer_subset flower102_tiny --output_dir output --prefer_mps` (tiny inference slice ≤2 batches). Metrics: top-1 accuracy + loss logged to TensorBoard; predictions to FiftyOne JSONL; labelmap emitted at `output/flower_eval/labelmap.json`.
- Datasets: flower102 is the smoke/default dataset with data_root supplied via env (e.g., `DATA_ROOT` from `.special_env`, git-ignored); other real datasets (CUB-200-2011, Cars, Dogs, NABirds, INat2017) remain optional with user-provided paths and licenses. Tiny subsets: `flower102_tiny` identifier for both train/eval; deterministic sampling (seed=42) limited to ≤2 batches per split.
- Smoke tests via pytest on synthetic tiny subsets (≤2 batches/split); no FP16/AMP; single-device (CPU/MPS). torch.distributed not used. TDD rule: add/adjust tests first, then implement.
- Output paths parameterized: `output/<run>/tb`, `output/<run>/fiftyone/predictions.jsonl`, `output/<run>/labelmap.json`, checkpoints under `output/<run>/checkpoints/ckpt.bin` (ignored by VCS), env stamp at `output/<run>/env_stamp.json`, retention note at `output/<run>/RETENTION.txt`.
- Variant axes: ViT-B/L/H patch sizes per existing configs; normalization from models/configs.py logged in metadata. No CLIP/SigLIP2/DINOv2 or linear-probe baselines in scope; scope-limited baseline table deferred for this minimal M2 pipeline (WAIVER: baseline table/linear probe out of scope for tiny smoke).
- Environment: conda/pip with Python 3.10+; env hash from `conda env export --from-history` (or pip freeze) stored in env_stamp. config_hash computed from resolved args/config; recorded in env_stamp. Requirements checksum not recorded.
- Label mapping: emit `output/<run>/labelmap.json` and reference in metadata; synthetic uses fixed index-order mapping; real datasets (when used) must log per-class metrics in TensorBoard/FiftyOne.
- Retention: TensorBoard/FiftyOne artifacts retained ≥90 days in `output/<run>/...`; retention note to be documented in PR/quickstart. Checkpoints not committed.
- Sanity eval: single-batch eval covered by smoke pytest; TDD enforced (tests first, then code). Tiny subset identifiers documented in research.md (flower102_tiny deterministic; no external checksum file).

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
LICENSE
README.md
requirements.txt
environment.yml (planned)
train.py
eval.py (dedicated eval-only CLI)
models/
  configs.py
  modeling.py
utils/
  autoaugment.py
  data_utils.py
  dataset.py
  dist_util.py
  scheduler.py
tests/
  (smoke tests to be added/updated for tiny train + eval)
specs/001-implement-base-pipeline/
  spec.md
  plan.md
  research.md
  data-model.md
  quickstart.md
  contracts/
  checklists/
```

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Baseline table / linear-probe per backbone deferred | Minimal Mac M2 smoke pipeline scope; synthetic tiny data not meaningful for linear-probe comparison | Running linear probes adds runtime/complexity without yielding useful metrics on synthetic smoke; will plan for real-data baselines in a later feature |
