# Tasks: Mac M2 Base Pipeline Implementation

**Input**: Design documents from /specs/001-implement-base-pipeline/
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are required per spec; smoke validations must be runnable via pytest on MacBook Air M2 (arm64, CPU/MPS).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and baseline hygiene

- [ ] T001 Update macOS arm64 stack pins in environment.yml and requirements.txt for PyTorch 2.x, torchvision 0.15+, ml_collections, tensorboard, pytest, optional fiftyone
- [ ] T002 Document smoke commands, pytest markers, and retention note steps in specs/001-implement-base-pipeline/quickstart.md to match current scope
- [ ] T003 Add output/* and output/*/checkpoints exclusions (no ckpt commits) to .gitignore
- [ ] T004 Add retention TTL note template to specs/001-implement-base-pipeline/checklists/requirements.md describing RETENTION.txt contents
- [ ] T005 Add a runnable FiftyOne readiness check script at scripts/check_fiftyone.py to validate import and headless app launch on macOS arm64
- [ ] T006 Document how to run the FiftyOne readiness check in specs/001-implement-base-pipeline/quickstart.md and note optional installation steps

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core utilities required by all user stories

- [ ] T007 Implement device selection helper with MPS-first CPU fallback and map_location support in utils/dist_util.py
- [ ] T008 Add deterministic seeding utility (seed all libs, worker init) in utils/data_utils.py
- [ ] T009 Create output path builder (tb, fiftyone, labelmap, checkpoints, retention) in utils/data_utils.py
- [ ] T010 Add labelmap writer for flower102 and other datasets with checksum support in utils/dataset.py
- [ ] T011 Add config hash/run_id helper in utils/dist_util.py (SHA256 of resolved args/config)

**Checkpoint**: Foundation ready - user story implementation can begin

---

## Phase 3: User Story 1 - Mac M2 tiny smoke pipeline (Priority: P1) 🎯 MVP

**Goal**: Runnable tiny train/infer smoke on MacBook Air M2 (MPS/CPU) with TensorBoard scalars and FiftyOne predictions.

**Independent Test**: Run `pytest -m "smoke_train or smoke" -q` then `DATA_ROOT=${DATA_ROOT:-/path/to/data} python train.py --name flower_mps --dataset flower102 --data_root "$DATA_ROOT"/flower102 --model_type testing --img_size 64 --num_steps 10 --eval_every 5 --train_batch_size 4 --eval_batch_size 2 --output_dir output --prefer_mps --tiny_train_subset flower102_tiny --tiny_infer_subset flower102_tiny` and confirm TensorBoard + predictions emitted.

### Tests for User Story 1 (TDD first)

- [ ] T012 [US1] Add smoke_train pytest covering tiny flower102 train/backward and export assertions in tests/test_smoke.py
- [ ] T013 [P] [US1] Add TensorBoard/FiftyOne contract test asserting prediction row count equals two eval batches (e.g., 4 rows at eval_batch_size=2) and labelmap presence for flower102 tiny subset in tests/test_smoke.py

### Implementation for User Story 1

- [ ] T014 [US1] Extend train.py CLI to accept flower102 tiny subset flags, data_root, prefer_mps, and output_dir with fp32-only defaults
- [ ] T015 [P] [US1] Integrate device selection + deterministic seeding from utils/dist_util.py and utils/data_utils.py inside train.py startup
- [ ] T016 [P] [US1] Emit TensorBoard scalars (loss/accuracy/lr) to output/tb/<run> in train.py
- [ ] T017 [US1] Write FiftyOne-compatible predictions JSONL and labelmap during eval_every in train.py using utils/data_utils.py helpers for flower102
- [ ] T018 [US1] Generate retention note output/<run>/RETENTION.txt recording TTL/location in train.py
- [ ] T019 [US1] Enforce fp32-only training (no AMP) and CPU fallback paths in train.py forward/backward and checkpoint save

**Checkpoint**: User Story 1 independently testable via smoke train command

---

## Phase 4: User Story 2 - Eval-only inference with predictions export (Priority: P2)

**Goal**: Dedicated eval-only CLI that loads checkpoint, runs validation, logs metrics, and writes predictions.

**Independent Test**: Run `pytest -m "smoke_eval or smoke" -q` then `DATA_ROOT=${DATA_ROOT:-/path/to/data} python eval.py --name flower_eval --dataset flower102 --data_root "$DATA_ROOT"/flower102 --img_size 64 --eval_batch_size 2 --checkpoint output/flower_mps/checkpoints/ckpt.bin --output_dir output --prefer_mps --tiny_infer_subset flower102_tiny` and confirm TensorBoard + predictions emitted.

### Tests for User Story 2 (TDD first)

- [ ] T020 [US2] Add smoke_eval pytest covering eval-only CLI outputs and row counts for flower102 tiny subset in tests/test_smoke.py
- [ ] T021 [P] [US2] Add missing-checkpoint fail-fast pytest covering map_location/device logging in tests/test_smoke.py

### Implementation for User Story 2

- [ ] T022 [US2] Extend eval.py CLI args for flower102 dataset/model/checkpoint/tiny_infer_subset/prefer_mps/output_dir handling
- [ ] T023 [P] [US2] Implement checkpoint load with fail-fast missing/invalid handling and map_location in eval.py
- [ ] T024 [P] [US2] Write FiftyOne predictions JSONL and labelmap for eval-only path in eval.py using shared helpers
- [ ] T025 [US2] Log eval metrics to output/tb/<run> in eval.py with parity to training logs

**Checkpoint**: User Story 2 independently testable via eval-only command

---

## Phase 5: User Story 3 - Reproducible environment and metadata (Priority: P3)

**Goal**: Reproducible setup with env_stamp metadata per run (train and eval) per constitution.

**Independent Test**: Create env via environment.yml (or pip requirements.txt), run `DATA_ROOT=${DATA_ROOT:-/path/to/data} python train.py --name meta_test --dataset flower102 --data_root "$DATA_ROOT"/flower102 --img_size 64 --num_steps 2 --eval_every 2 --train_batch_size 2 --eval_batch_size 2 --tiny_train_subset flower102_tiny --tiny_infer_subset flower102_tiny --output_dir output --prefer_mps` and verify env_stamp.json fields are populated.

### Tests for User Story 3 (TDD first)

- [ ] T026 [P] [US3] Add env_stamp field completeness pytest covering required keys and fp16=false in tests/test_smoke.py

### Implementation for User Story 3

- [ ] T027 [P] [US3] Implement env_stamp builder capturing required fields (config_hash, env_hash, device info, package versions, fp16=false) in utils/dist_util.py
- [ ] T028 [US3] Write env_stamp.json during training runs in train.py using builder and run args
- [ ] T029 [US3] Write env_stamp.json during eval-only runs in eval.py capturing checkpoint and tiny subset info
- [ ] T030 [P] [US3] Document env hash export/checksum commands in specs/001-implement-base-pipeline/quickstart.md and ensure TDD markers reference them
- [ ] T031 [P] [US3] Add requirements/env hash validation helper (reads env_export.yml or pip freeze) to utils/dist_util.py and include in env_stamp

**Checkpoint**: User Story 3 independently testable via env_stamp inspection

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Hardening and doc alignment

- [ ] T032 Validate quickstart.md commands end-to-end (train + eval) and update README.md snippets accordingly
- [ ] T033 Clean up logging/error messages for missing data_root/checkpoint and ensure help text matches contracts/openapi.yaml in train.py and eval.py

---

## Dependencies & Execution Order

- Setup (Phase 1) → Foundational (Phase 2) → US1 (P1) → US2 (P2) → US3 (P3) → Polish
- US1 is MVP and must complete before US2/US3 to provide checkpoint and helpers
- Foundational tasks block all user stories; Polish waits on stories chosen for delivery

## User Story Dependency Graph

US1 (P1) → US2 (P2) → US3 (P3)

## Parallel Opportunities (examples)

- Foundational: T007, T008, T009, T010, T011 can proceed in parallel after T001–T006
- US1: T015 and T016 can run in parallel after T014; T017 depends on T009/T010 completion
- US2: T023 and T024 can run in parallel after T022; T025 after logging helper available
- US3: T027 and T031 can run in parallel; T028/T029 follow T027

## Implementation Strategy

- MVP first: finish Setup → Foundational → US1, run smoke_train pytest + train CLI to validate
- Incremental: add US2 eval-only once US1 delivers checkpoints; add US3 env_stamp metadata last
- Each user story stays independently testable with its smoke command and pytest marker
