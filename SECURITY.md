# Security policy

## Credentials

Do not commit `.env`, API keys, tokens, participant records, resumes, job
descriptions, or raw interview audio. Copy `.env.example` to `.env` locally.

If a live key has ever been committed, deleting the latest copy is not enough:
rotate/revoke the key immediately and follow GitHub's documented sensitive-data
removal process if history must be rewritten.

## Reporting

Report suspected credential exposure privately to the repository owner. Do not
open a public issue containing a credential or participant information.

## Research data

Only de-identified derived evidence whose approval and consent basis permits
release may enter the public reproducibility archive. The validation scripts
reject common direct-identifier columns, but automated checks do not replace
human disclosure review.
