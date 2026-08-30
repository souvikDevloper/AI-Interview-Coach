# Reproducibility status

## Implemented and testable

- Mode-adaptive BM25 + dense retrieval with paper-matched weights.
- Deterministic 500-token-like chunks with 50-token overlap.
- Stable chunk provenance and score components.
- Cluster-level paired bootstrap confidence intervals.
- Exact/Monte-Carlo sign-flip randomisation tests.
- Independent-session cluster bootstrap, label-permutation tests, and Hedges' g.
- Micro/session-macro WER and survey reliability/group-summary scripts.
- Evidence schemas, range/duplicate/identifier validation, and SHA-256 manifests.
- Versioned LoRA hyperparameter configuration and corpus validation gate.
- Deterministic unit tests and CI on Python 3.11/3.12.

## Not included in this repository

- Original participant-level or item-level human-study exports.
- Participant resumes, job descriptions, audio, or signed consent forms.
- Fine-tuned model weights or the underlying 500 interview-transcript pairs.
- A confirmatory clustered result computed from the original study data.

These omissions are deliberate until the authors confirm provenance,
de-identification, approval scope, and redistribution rights. Synthetic fixtures
under `tests/fixtures/` verify software behavior only and must never be cited as
study evidence.
