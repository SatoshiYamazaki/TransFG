# Feature Specification: Mac M2 Base Pipeline Implementation

**Feature Branch**: `001-implement-base-pipeline`  
**Created**: 2026-01-19  
**Status**: Draft  
**Input**: User description: "Implement code and tests for Mac M2 base pipeline (MPS/CPU), pytest smoke, TensorBoard/FiftyOne logging, conda env, inference entrypoint"

## User Scenarios & Testing *(mandatory)*

Tests and smoke validations MUST be runnable via pytest (e.g., markers for tiny train/infer subsets ≤2 batches/split). Document the exact pytest commands expected to exercise each user story on the MacBook Air M2 (arm64, CPU/MPS) environment.

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Implement Mac M2 tiny smoke pipeline (Priority: P1)

Ship runnable tiny train/infer smoke on MacBook Air M2 (MPS or CPU fallback) with TensorBoard scalars and FiftyOne-compatible predictions, backed by code and pytest smoke tests.

**Why this priority**: Establishes working baseline and governance compliance (tiny subsets, pytest, logging) on the primary dev environment.

**Independent Test**: Run `pytest tests/test_smoke.py -q` then execute `DATA_ROOT=${DATA_ROOT:-/path/to/data} python train.py --name flower_mps --dataset flower102 --data_root "$DATA_ROOT"/flower102 --model_type testing --img_size 64 --num_steps 10 --eval_every 5 --train_batch_size 4 --eval_batch_size 2 --tiny_train_subset flower102_tiny --tiny_infer_subset flower102_tiny --output_dir output --prefer_mps` (fp32 only; AMP omitted) and confirm artifacts emit without CUDA. The flower102 tiny subset (≤2 batches/split) is the smoke dataset.

**Acceptance Scenarios**:

1. **Given** the flower102 tiny subset and updated train.py, **When** pytest smoke runs, **Then** forward/backward and validation export tests pass under 3 minutes on MPS/CPU.
2. **Given** the smoke command above, **When** it completes, **Then** TensorBoard logs exist under `output/tb/flower_mps` and `output/flower_mps/fiftyone/predictions.jsonl` contains the expected rows for two eval batches (e.g., 4 rows with `eval_batch_size=2`).

---

### User Story 2 - Eval-only inference with predictions export (Priority: P2)

Enable a dedicated eval-only CLI (separate from the training entrypoint) that loads a checkpoint, runs validation without training, logs metrics, and writes FiftyOne predictions.

**Why this priority**: Supports quick regression checks and artifact generation without running full training.

**Independent Test**: Run `DATA_ROOT=${DATA_ROOT:-/path/to/data} python eval.py --name flower_eval --dataset flower102 --data_root "$DATA_ROOT"/flower102 --pretrained_dir /path/to/ViT-B_16.npz --checkpoint output/flower_mps/checkpoints/ckpt.bin --tiny_infer_subset flower102_tiny --eval_batch_size 2` and verify outputs. Eval-only must, at minimum, support a tiny eval slice (≤2 batches) completing in ≤5 minutes on MPS/CPU; larger validations are allowed when resources/time permit.

**Acceptance Scenarios**:

1. **Given** a valid checkpoint and the flower102 dataset path, **When** running the eval-only command, **Then** the run completes on MPS/CPU, writes TensorBoard scalars to `output/tb/flower_eval`, and creates `output/flower_eval/fiftyone/predictions.jsonl` with one row per evaluated sample from the tiny subset (e.g., 4 rows with two eval batches of size 2).

---

### User Story 3 - Reproducible environment and metadata (Priority: P3)

Deliver reproducible setup (conda/pip) plus env_stamp metadata per run per constitution.

**Why this priority**: Guarantees traceability and governance compliance across dev/CI.

**Independent Test**: Create env via `conda env create -f environment.yml` (or `pip install -r requirements.txt`), run `DATA_ROOT=${DATA_ROOT:-/path/to/data} python train.py --name meta_test --dataset flower102 --data_root "$DATA_ROOT"/flower102 --img_size 64 --num_steps 2 --eval_every 2 --train_batch_size 2 --eval_batch_size 2 --tiny_train_subset flower102_tiny --tiny_infer_subset flower102_tiny --output_dir output --prefer_mps` (training-only, fp32), and inspect generated env_stamp.json. Use the dedicated eval CLI separately when validating checkpoints.

**Acceptance Scenarios**:

1. **Given** the env setup and meta_test command, **When** the run completes, **Then** `output/meta_test/env_stamp.json` contains all required fields with non-empty values.
2. **Given** the same run, **When** inspecting CLI logs, **Then** device selection (MPS or CPU) is logged.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- MPS unavailable: pipeline must auto-fallback to CPU without CUDA dependencies and still pass smoke/eval commands.
- Missing pretrained weights: training can proceed with random init and a warning; eval-only MUST fail fast with a clear error if `--checkpoint` is absent or unreadable.
- Missing data_root for real datasets: fail fast with actionable error before partial outputs are written.
- Missing optional packages (e.g., FiftyOne): env_stamp MUST include version when installed; if absent, the run MUST warn and continue while recording `fiftyone_version: "missing"`.

## Requirements *(mandatory)*

### Constitution Alignment

- pytest MUST be the test runner; tiny train/infer subsets ≤2 batches/split on flower102 must be runnable on Mac M2 (MPS/CPU) and codified in tests/commands.
- env stamp JSON with canonical schema must be produced per run; TensorBoard and FiftyOne artifacts must be generated and referenced.
- Commands/configs must run on macOS arm64 without CUDA assumptions; distributed launch is optional/off by default.

### Functional Requirements

- **FR-001**: Implement and maintain pytest smoke tests covering forward/backward and validation export; they must run in ≤3 minutes on Mac M2 (MPS/CPU) using flower102 tiny subsets.
- **FR-002**: Implement a Mac M2-friendly tiny training command (fp32, no AMP) on flower102 that emits TensorBoard scalars under `output/tb/<run>` and FiftyOne JSONL predictions under `output/<run>/fiftyone/predictions.jsonl` with row count equal to evaluated flower102 samples (e.g., 4 rows for two eval batches of size 2); emit `output/<run>/labelmap.json` and log per-class metrics to TensorBoard/FiftyOne.
- **FR-003**: Implement a dedicated eval-only CLI (separate entrypoint from training) to load checkpoints, run validation without training, and emit TensorBoard + FiftyOne artifacts on flower102 MPS/CPU; emit `output/<run>/labelmap.json` and log per-class metrics to TensorBoard/FiftyOne.
- **FR-004**: Auto-detect device preferring MPS with CPU fallback; log device selection; no CUDA dependency may block runs on macOS.
- **FR-005**: Write env_stamp.json per run containing required fields (run_id, config_hash, git_commit, os_version, python_version, torch_version, torchvision_version, ml_collections_version, driver_version, cuda_version, cudnn_version, gpu_name, gpu_count, fp16 set to false, seed, env_hash, tiny_train_subset_path, tiny_infer_subset_path, timestamp, fiftyone_version, torch_build) stored under `output/<run>/env_stamp.json`.
- **FR-006**: Provide reproducible env artifacts: updated environment.yml and requirements.txt; env stamp records env_hash but not requirements checksum.
- **FR-007**: Handle missing pretrained weights gracefully with warnings and random init for training; eval-only MUST fail fast if checkpoint is missing or unreadable; checkpoint loading must honor map_location for selected device.
- **FR-008**: Expose CLI options for tiny subset paths, data_root, checkpoint, FiftyOne output path, and MPS preference to satisfy governance declarations.
- **FR-009**: Enforce artifact retention metadata: TensorBoard and FiftyOne outputs MUST declare retention ≥90 days and their storage location in PRs/readme snippets; default schema is `output/{run}/tb` and `output/{run}/fiftyone` unless explicitly overridden and documented. Record retention evidence (e.g., RETENTION.txt) alongside run outputs.

### Key Entities *(include if feature involves data)*

- **Env Stamp**: JSON document capturing run metadata (fields in FR-005) stored at `output/<run>/env_stamp.json`.
- **Prediction Artifact**: JSONL file compatible with FiftyOne containing per-sample `sample_id`, `prediction`, and `label`, stored at `output/<run>/fiftyone/predictions.jsonl`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Automated pytest smoke suite completes in ≤3 minutes on MacBook Air M2 (MPS/CPU) with zero failures.
- **SC-002**: Tiny flower102 training run produces visible loss/lr/accuracy metrics in logging artifacts and a predictions file whose row count equals evaluated samples (e.g., 4 rows for two eval batches of size 2 on the tiny subset).
- **SC-003**: Eval-only run (via the dedicated eval CLI) on flower102 completes on a non-CUDA device, processes the tiny slice (≤2 batches) in ≤5 minutes, produces validation metrics, logs per-class metrics, and emits a predictions file whose row count matches the evaluation dataset size with no missing/extra rows; larger validations MAY be run when resources allow.
- **SC-004**: Each run writes env_stamp.json containing all required fields (including fiftyone_version and torch_build, fp16 set to false) with non-empty values.
- **SC-005**: Device selection is recorded as MPS when available and CPU otherwise; runs do not fail due to absence of CUDA on macOS.
- **SC-006**: Retention metadata for TensorBoard and FiftyOne artifacts is documented (≥90 days) alongside their storage paths.

## Assumptions

- The flower102 dataset is the smoke dataset; the data root is supplied via env (e.g., `DATA_ROOT` set in a git-ignored `.special_env` file) and tiny subsets (≤2 batches) are required for smoke.
- Users provide valid data_root and checkpoint paths when invoking eval-only on real datasets.
- Internet access is available to install dependencies via conda/pip on macOS arm64.
