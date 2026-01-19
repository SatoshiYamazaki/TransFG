---

description: "Task list template for feature implementation"
---

# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification. When included, tests and smoke validations MUST be runnable via pytest on MacBook Air M2 (arm64, CPU/MPS).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

<!-- 
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.
  
  The /speckit.tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/
  
  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment
  
  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize [language] project with [framework] dependencies
- [ ] T003 [P] Configure linting and formatting tools
- [ ] T004 Document dataset source and license (from allowed public datasets) in specs/[###-feature]/plan.md
- [ ] T005 Capture baseline training/eval command, config path, seeds, and output directories for reproducibility
- [ ] T006 Create pytest smoke-test targets that run ≤2 batches per split (respect FP16/distributed flags) using the tiny training subset and tiny inference subset; keep paths and checksums documented in spec; commands MUST run on MacBook Air M2 (CPU/MPS, no CUDA assumed)
- [ ] T007 Define variant matrix (ViT size, backbone: CLIP/SigLIP2/DINOv2, part attention on/off, patch/resolution) mapped to configs; block modern backbones unless upgraded stack (Python/PyTorch/extra libs), README/requirements updates, and a recorded TensorBoard+FiftyOne validation command/output are declared and checked into the repo (script/target)
- [ ] T008 Document preprocessing/normalization choices per backbone to ensure fair comparisons
- [ ] T009 Plan linear-probe baselines per backbone (frozen encoder, linear head) with commands/configs and reporting target
- [ ] T010 Execute and record linear-probe baselines before/alongside finetune runs
- [ ] T011 Capture miniconda environment export (environment.yml or conda env export) for reproducibility, store env hash and requirements.txt checksum
- [ ] T012 Configure TensorBoard logging path/keys and ensure runs write scalars
- [ ] T013 Define storage path/format for FiftyOne-ready inference artifacts and link to checkpoints/configs
- [ ] T014 Define stable output path schema (e.g., outputs/{run_id}/{config_hash}/{checkpoint_id}/tb and /fiftyone) and apply to planned runs
- [ ] T015 Specify baseline metrics artifact format (CSV/JSON) with required fields and storage path
- [ ] T016 Plan label mapping/order validation steps for each dataset/backbone combo
- [ ] T017 List system/env metadata to capture per run (os_version, git_commit, config_hash, GPU/MPS model/count, CUDA/cuDNN or MPS driver, key package versions)
- [ ] T018 Record env hash from conda export and store alongside run metadata, with requirements.txt checksum; define config_hash as SHA256 of resolved config contents, store env stamp at the metadata path declared in spec, and cite in PRs
- [ ] T018 Record env hash from conda export and store alongside run metadata, with requirements.txt checksum; define config_hash as SHA256 of resolved config contents, store env stamp JSON with canonical fields (run_id, config_hash, git_commit, os_version, python_version, torch_version, torchvision_version, ml_collections_version, driver_version, cuda_version, cudnn_version, gpu_name, gpu_count, fp16, seed, requirements_checksum, env_hash, tiny_train_subset_path, tiny_infer_subset_path) at the metadata path declared in spec, and cite in PRs with an excerpt
- [ ] T019 Define retention/TTL and required contents for TensorBoard/FiftyOne artifacts (per-class metrics, confusion matrix, top-k, misclassifications) and how retention evidence/TTL will be recorded in PRs (TTL policy link or storage path with expiry note)
- [ ] T020 Plan minimal eval sanity check (single-batch eval), ensure tests are added/updated before implementation (TDD) as pytest tests, and include tiny train/inference smoke commands with expected runtime and label coverage (runnable on MacBook Air M2 CPU/MPS)
- [ ] T021 Document approval/license review if introducing new/synthetic datasets
- [ ] T022 Record requirements.txt checksum and include python_version and ml_collections_version in metadata plan
- [ ] T023 Ensure baseline artifact includes per-class metrics (e.g., macro-F1 or per-class accuracy)
- [ ] T024 Define label-map artifact path (outputs/{run_id}/{config_hash}/{checkpoint_id}/labelmap.json) and checksum logging
- [ ] T025 Set minimum retention window (≥90 days) for TensorBoard/FiftyOne artifacts and baseline tables (unless release policy requires longer)
- [ ] T026 Define checkpoint handling: specify external/ignored storage path schema outputs/{run_id}/{config_hash}/{checkpoint_id}/ckpt.bin (or equivalent), ensure checkpoints are not committed, and record SHA256 checkpoint file hash referenced by checkpoint_id

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [ ] T026 Setup database schema and migrations framework
- [ ] T027 [P] Implement authentication/authorization framework
- [ ] T028 [P] Setup API routing and middleware structure
- [ ] T029 Create base models/entities that all stories depend on
- [ ] T030 Configure error handling and logging infrastructure
- [ ] T031 Setup environment configuration management

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - [Title] (Priority: P1) 🎯 MVP

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T032 [P] [US1] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T033 [P] [US1] Integration test for [user journey] in tests/integration/test_[name].py

### Implementation for User Story 1

- [ ] T034 [P] [US1] Create [Entity1] model in src/models/[entity1].py
- [ ] T035 [P] [US1] Create [Entity2] model in src/models/[entity2].py
- [ ] T036 [US1] Implement [Service] in src/services/[service].py (depends on T034, T035)
- [ ] T037 [US1] Implement [endpoint/feature] in src/[location]/[file].py
- [ ] T038 [US1] Add validation and error handling
- [ ] T039 [US1] Add logging for user story 1 operations

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - [Title] (Priority: P2)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [ ] T040 [P] [US2] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T041 [P] [US2] Integration test for [user journey] in tests/integration/test_[name].py

### Implementation for User Story 2

- [ ] T042 [P] [US2] Create [Entity] model in src/models/[entity].py
- [ ] T043 [US2] Implement [Service] in src/services/[service].py
- [ ] T044 [US2] Implement [endpoint/feature] in src/[location]/[file].py
- [ ] T045 [US2] Integrate with User Story 1 components (if needed)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - [Title] (Priority: P3)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [ ] T046 [P] [US3] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T047 [P] [US3] Integration test for [user journey] in tests/integration/test_[name].py

### Implementation for User Story 3

- [ ] T048 [P] [US3] Create [Entity] model in src/models/[entity].py
- [ ] T049 [US3] Implement [Service] in src/services/[service].py
- [ ] T050 [US3] Implement [endpoint/feature] in src/[location]/[file].py

**Checkpoint**: All user stories should now be independently functional

---

[Add more user story phases as needed, following the same pattern]

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] TXXX [P] Documentation updates in docs/
- [ ] TXXX Code cleanup and refactoring
- [ ] TXXX Performance optimization across all stories
- [ ] TXXX [P] Additional unit tests (if requested) in tests/unit/
- [ ] TXXX Security hardening
- [ ] TXXX Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

- ### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
Task: "Contract test for [endpoint] in tests/contract/test_[name].py"
Task: "Integration test for [user journey] in tests/integration/test_[name].py"

# Launch all models for User Story 1 together:
Task: "Create [Entity1] model in src/models/[entity1].py"
Task: "Create [Entity2] model in src/models/[entity2].py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
