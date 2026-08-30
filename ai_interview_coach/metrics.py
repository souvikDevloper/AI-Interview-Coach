"""Dependency-free word-error and survey reliability metrics."""

from __future__ import annotations

import re
from collections import defaultdict
from math import fsum
from statistics import variance
from typing import Iterable, Mapping


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return Levenshtein word errors and reference-word count."""
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            ))
        previous = current
    return previous[-1], len(ref)


def cronbach_alpha(rows: Iterable[Mapping[str, object]]) -> float:
    """Compute alpha for complete participant-by-item response records."""
    participants: dict[str, dict[str, float]] = defaultdict(dict)
    all_items: set[str] = set()
    for row in rows:
        participant = str(row["participant_id"])
        item = str(row["item_id"])
        participants[participant][item] = float(row["response"])
        all_items.add(item)
    if len(all_items) < 2 or len(participants) < 2:
        raise ValueError("Cronbach's alpha requires at least two items and two participants")
    incomplete = [participant for participant, values in participants.items() if set(values) != all_items]
    if incomplete:
        raise ValueError("Cronbach's alpha requires complete item responses per participant")
    ordered_items = sorted(all_items)
    item_variances = [
        variance([values[item] for values in participants.values()]) for item in ordered_items
    ]
    total_scores = [fsum(values[item] for item in ordered_items) for values in participants.values()]
    total_variance = variance(total_scores)
    if total_variance == 0:
        raise ValueError("Cronbach's alpha is undefined when total scores have zero variance")
    k = len(ordered_items)
    return k / (k - 1.0) * (1.0 - fsum(item_variances) / total_variance)
