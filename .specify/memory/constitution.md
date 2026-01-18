<!--
Sync Impact Report
- Version change: 1.2.0 -> 1.3.0
- Modified principles: Reproducible Experiments; Variant Traceability & Fair Comparisons; Baseline Fidelity; Evaluation Fidelity; Runtime Reliability & Minimal Testing; Experiment Logging & Visibility
- Added sections: none
- Removed sections: none
- Templates requiring updates: ✅ .specify/templates/plan-template.md; ✅ .specify/templates/spec-template.md; ✅ .specify/templates/tasks-template.md
- Follow-up TODOs: none
-->
# TransFG Constitution

## Core Principles

### Reproducible Experiments (NON-NEGOTIABLE)
Every experiment MUST be reproducible from repository state plus a committed config in models/configs.py. Training commands with all flags, random seeds, dataset splits, and output checkpoints MUST be recorded alongside results. Changes that affect determinism (augmentation, data shuffling, mixed precision) MUST note their impact and seed handling in the plan/spec.

### Data Provenance & Licensing
Only publicly listed datasets in README (CUB-200-2011, Stanford Cars, Stanford Dogs, NABirds, iNaturalist2017) MAY be used. Dataset source, license terms, and any filtering or relabeling MUST be documented before training. Private or user-supplied data MUST NOT be committed; paths must stay configurable and out of version control.

### Configuration & Change Isolation
Hyperparameters and architecture tweaks MUST live in configs (ml_collections) with safe defaults. Each change set MUST isolate one behavioral change per PR (e.g., new augment policy, optimizer setting) and include rationale plus expected effect. Hidden defaults in scripts are prohibited; every runtime knob must be discoverable in config or CLI flags.

### Variant Traceability & Fair Comparisons
Backbone choice (e.g., ViT-B/L/H from CLIP, SigLIP2, DINOv2), part attention toggles, and patch/resolution settings MUST be declared in configs with clear names. Comparisons MUST hold preprocessing, dataset splits, and evaluation metrics constant; when they differ, the impact MUST be justified and logged. Results MUST reference the exact pretrained weights (source/commit) and normalization used per backbone.

### Baseline Fidelity (Linear Probe)
For each backbone family (CLIP, SigLIP2, DINOv2), a frozen-encoder linear-probe baseline MUST be run and reported before or alongside finetuning. Commands, configs, feature-extraction resolution, and normalization must be recorded. Finetune claims MUST cite the paired linear-probe result for the same backbone and preprocessing.

### Experiment Logging & Visibility
All training runs MUST emit TensorBoard logs (loss/accuracy curves and key scalars) to a committed-relative path. Inference outputs generated during training (e.g., val/test predictions) MUST be persisted in a format consumable by FiftyOne, including labels, logits/scores, and metadata for backbone, config, and checkpoint. Log and artifact paths MUST be recorded in plan/spec and surfaced in PRs.

### Evaluation Fidelity
Evaluation MUST use the canonical splits and image resolutions for each dataset. Report at minimum top-1 accuracy and loss; note evaluation batch size, checkpoint used, and any test-time augmentation. Benchmarks MUST cite the reference paper/commit they compare against and avoid mixing incompatible preprocessing or label spaces.

### Runtime Reliability & Minimal Testing
Before long runs, a smoke test MUST execute a tiny subset (e.g., 2 batches per split) to validate data loading, model construction, and forward/backward passes (FP16 if used). Training runs MUST specify GPU count, distributed launch parameters, and resource bounds to avoid cluster contention. Failures MUST capture logs with the command and config used.

## Operational Constraints
Target stack: Python 3.7.x, PyTorch 1.5.1, torchvision 0.6.1, ml_collections. Training is GPU-bound and assumes torch.distributed launch when nproc_per_node > 1; single-GPU paths MUST remain working. Keep requirements.txt authoritative; do not vendor weights or datasets into the repo. FP16 is optional but, if enabled, must include loss-scaling settings. Backbones may use official weights from CLIP, SigLIP2, or DINOv2; record source/commit for each and never commit checkpoints. Linear-probe runs should use frozen encoders and lean batch sizes to validate data/feature pipelines quickly. Each experiment MUST be runnable in a documented miniconda environment (environment.yml or export). TensorBoard logging and FiftyOne-compatible inference artifacts MUST be written to configurable output dirs. Data paths and output directories MUST be parameterized, never hard-coded.

## Development Workflow
Use feature branches per change. Every PR MUST include: (a) training/eval command(s) and config path, (b) dataset provenance note if data handling changes, (c) smoke-test evidence or rationale if skipped, and (d) recorded metrics/log location. Documentation updates (README snippets or per-feature quickstart) MUST accompany new configs or flags. No merge if Constitution Check gates in plan.md/tasks.md fail.

## Governance
This constitution supersedes other process docs when in conflict. Amendments require a PR citing the change, updated version, and rationale; at least one maintainer must approve. Versioning follows semantic rules: MAJOR for breaking governance/principle removals, MINOR for added or expanded principles/sections, PATCH for clarifications. Compliance is reviewed during planning (Constitution Check), pre-merge reviews, and before releasing new checkpoints.

**Version**: 1.3.0 | **Ratified**: 2026-01-19 | **Last Amended**: 2026-01-19
