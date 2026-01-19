# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

[Extract from feature spec: primary requirement + technical approach from research]

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: [e.g., Python 3.11, Swift 5.9, Rust 1.75 or NEEDS CLARIFICATION]  
**Primary Dependencies**: [e.g., FastAPI, UIKit, LLVM or NEEDS CLARIFICATION]  
**Storage**: [if applicable, e.g., PostgreSQL, CoreData, files or N/A]  
**Testing**: [pytest REQUIRED; add markers/commands]  
**Target Platform**: [macOS (MacBook Air M2, arm64, CPU/MPS); note if other targets]
**Project Type**: [single/web/mobile - determines source structure]  
**Performance Goals**: [domain-specific, e.g., 1000 req/s, 10k lines/sec, 60 fps or NEEDS CLARIFICATION]  
**Constraints**: [domain-specific, e.g., <200ms p95, <100MB memory, offline-capable or NEEDS CLARIFICATION]  
**Scale/Scope**: [domain-specific, e.g., 10k users, 1M LOC, 50 screens or NEEDS CLARIFICATION]

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Record exact training/eval command(s), config path, seeds, and output directories.
- Confirm dataset source + license (CUB-200-2011, Stanford Cars/Dogs, NABirds, iNaturalist2017) and any filtering steps.
- Plan smoke test on a tiny subset (2 batches/split) including FP16/distributed params if used; smoke MUST be runnable via pytest.
- Define evaluation split, resolution, and metrics (top-1 accuracy + loss) tied to the planned checkpoint.
- Declare GPU/MPS count, torch.distributed launch shape (if any), and ensure data/output paths are parameterized (not hard-coded); commands MUST run on MacBook Air M2 (arm64 CPU/MPS) without CUDA assumptions.
- Enumerate variant axes (ViT size, backbone: CLIP/SigLIP2/DINOv2, part attention on/off, patch/resolution) and how comparability will be enforced (shared preprocessing, normalization source, eval metric). Modern backbones (CLIP/SigLIP2/DINOv2) MUST NOT proceed unless the plan declares the upgraded stack (Python/PyTorch/extra libs), README/requirements updates, and a recorded validation command/output (checked-in script/target) proving TensorBoard and FiftyOne run on that stack.
- Include linear-probe baseline plan per backbone (frozen encoder, linear head): commands, configs, normalization, and reporting alongside finetune runs.
- Provide miniconda environment details (environment.yml or export) and how it is applied in runs. Specify the tiny training subset and tiny inference subset used for smoke tests (e.g., ≤2 batches per split), where they live, expected runtime, label coverage, and checksum if applicable.
- Specify TensorBoard logging location and keys to capture; define where FiftyOne-ready inference artifacts will be written and how they link to checkpoints/configs.
- Define stable output path schema (e.g., outputs/{run_id}/{config_hash}/{checkpoint_id}/tb and /fiftyone) for logs/artifacts.
- Capture normalization identifier per backbone and how it is logged in artifacts.
- Plan baseline metrics artifact (CSV/JSON) fields: backbone, config name, normalization ID, resolution, head size, seed, checkpoint/hash, top-1/etc., and linkage to finetune runs.
- State how label mapping/order will be validated against dataset definitions.
- Note system/env metadata to record (os_version, git_commit, config_hash, GPU model/count, CUDA/cuDNN/driver, key package versions). 
- Require env hash from `conda env export --from-history` (or equivalent) stored with run metadata, plus requirements.txt checksum; compute config_hash as SHA256 of resolved config contents (after interpolation) and record where the env stamp JSON (canonical schema) will be logged (path declared in spec) and cited in PRs, including a sample excerpt.
- Define retention/TTL for TensorBoard/FiftyOne artifacts and minimum contents (per-class accuracy, confusion matrix, top-k metrics, misclassification view), and record retention evidence (TTL policy link or storage path with expiry note) in the PR.
- Plan minimal evaluation sanity check (single-batch eval) and enforce TDD (tests added/updated before implementation) plus required tiny train/inference smoke commands.
- If introducing new/synthetic datasets, document approval path and license review.
- Require baseline artifacts to include per-class metrics (e.g., macro-F1 or per-class accuracy).
- Specify label-map artifact path schema (outputs/{run_id}/{config_hash}/{checkpoint_id}/labelmap.json) and checksum capture.
- Define checkpoint handling: checkpoints are never committed; specify external/ignored storage path schema outputs/{run_id}/{config_hash}/{checkpoint_id}/ckpt.bin (or equivalent) and record SHA256 hash to reference via checkpoint_id.
- Set minimum retention window (≥90 days) for TensorBoard/FiftyOne artifacts and baseline tables unless tied to longer-lived releases.
- State adherence to the env stamp JSON schema and where the file will live (path), plus required fields (run_id, config_hash, git_commit, os_version, python_version, torch_version, torchvision_version, ml_collections_version, driver_version, cuda_version, cudnn_version, gpu_name, gpu_count, fp16, seed, requirements_checksum, env_hash, tiny_train_subset_path, tiny_infer_subset_path).

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
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
