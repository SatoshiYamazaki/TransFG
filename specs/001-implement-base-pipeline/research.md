# Research (Phase 0)

## PyTorch on MPS (best practices)
- Decision: Prefer `mps` when available; keep fp32 only, small batch sizes (≤4 train / ≤2 eval) and `num_workers<=2`; avoid pinned memory and autocast.
- Rationale: M2 GPU memory is limited and lacks CUDA AMP; smaller batches reduce OOM risk and keep smoke under 3–5 minutes.
- Alternatives considered: Use CUDA AMP (not available on Mac M2) or CPU-only fallback for all runs (would slow smoke unnecessarily).

## Pytest marker naming (clarification)
- Decision: Use two markers: `smoke_train` for tiny train/backward and `smoke_eval` for eval-only inference; a combined `smoke` marker may select both.
- Rationale: Separation keeps TDD aligned to distinct user stories (train vs eval) while allowing a single aggregated run for CI (`-m smoke`).
- Alternatives considered: Single `smoke` marker only (less granular) or dataset-specific markers (over-segmentation for this baseline).

## Retention evidence path (clarification)
- Decision: Document retention ≥90 days in quickstart and PRs; store a text note alongside artifacts at `output/<run>/RETENTION.txt` stating TTL and storage path, and cite this file in PRs.
- Rationale: Lightweight evidence satisfies constitution without external policy links; colocating the note prevents drift between code and docs.
- Alternatives considered: External policy URL (may drift) or embedding TTL only in README (not per-run, harder to audit).

## Env hash capture (best practices)
- Decision: Capture env hash via `conda env export --from-history > env_export.yml` (or `pip freeze > requirements-lock.txt` fallback) and hash that export; record hash plus requirements.txt checksum in env_stamp.
- Rationale: From-history exports reduce transient deps and keep hashes stable; fallback covers pip-only environments.
- Alternatives considered: Full `conda env export` with explicit builds (noisier, less stable) or skipping env hash (violates constitution).
