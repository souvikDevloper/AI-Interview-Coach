# Reproducibility archive

This directory defines the release contract for the IEEE TLT study. It does
not contain fabricated observations. Add only de-identified records whose
provenance and consent basis have been confirmed under the college approval.

## Required private-to-public workflow

1. Export original item-level records using the header-only templates in `schemas/`.
2. Replace direct identifiers with stable study IDs. Do not publish names,
   emails, roll numbers, phone numbers, resumes, job descriptions, or raw audio
   unless approval and participant consent explicitly permit redistribution.
3. Run `python scripts/validate_evidence.py <schema> <file.csv>` for each file.
4. Run the cluster-aware analysis, for example:
   `python scripts/analyze_clustered.py retrieval_items.csv --treatment D --control A --output clustered-results.json`
   Use `scripts/analyze_misc_clustered.py` for independent session groups,
   `scripts/analyze_wer.py` for micro and session-macro WER, and
   `scripts/analyze_survey.py` for scale reliability and group descriptives.
5. Record package versions, Git commit, hardware, random seed, and command.
6. Generate `evidence-manifest.json` with `scripts/build_evidence_manifest.py`.
7. Deposit the approved archive in a versioned repository and cite its DOI.

## Statistical unit

Retrieval questions are nested within source resumes/job descriptions, and
MISC utterances are nested within sessions. Confirmatory inference must operate
on independent clusters (or use a justified mixed-effects model), not treat
every nested item as independent. The paired script requires each retrieval
source to occur in both configurations; the independent MISC script rejects
session IDs that cross systems, preventing accidental use of the wrong design.

## Public-data boundary

When source documents or audio cannot be redistributed, publish derived,
non-identifying item-level measures plus a data dictionary, exclusion log,
analysis scripts, seeds, and a transparent controlled-access procedure.
