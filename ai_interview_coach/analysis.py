"""Cluster-aware paired analysis for nested retrieval and session evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
import random
from statistics import mean, stdev
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ClusterPairedResult:
    metric: str
    treatment: str
    control: str
    clusters_used: int
    clusters_dropped: int
    treatment_mean: float
    control_mean: float
    mean_difference: float
    percent_change: float | None
    ci_low: float
    ci_high: float
    paired_standardized_effect: float | None
    sign_flip_p_value: float
    bootstrap_replicates: int
    permutation_replicates: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClusterIndependentResult:
    metric: str
    treatment: str
    control: str
    treatment_clusters: int
    control_clusters: int
    treatment_mean: float
    control_mean: float
    mean_difference: float
    percent_change: float | None
    ci_low: float
    ci_high: float
    hedges_g: float | None
    permutation_p_value: float
    bootstrap_replicates: int
    permutation_replicates: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sign_flip_p_value(differences: list[float], *, rng: random.Random, replicates: int) -> tuple[float, int]:
    observed = abs(mean(differences))
    n = len(differences)
    if n <= 20:
        values = [
            abs(mean([sign * difference for sign, difference in zip(pattern, differences)]))
            for pattern in itertools.product((-1.0, 1.0), repeat=n)
        ]
        return sum(value >= observed - 1e-12 for value in values) / len(values), len(values)
    exceedances = 0
    for _ in range(replicates):
        permuted = mean([difference * rng.choice((-1.0, 1.0)) for difference in differences])
        exceedances += abs(permuted) >= observed - 1e-12
    return (exceedances + 1.0) / (replicates + 1.0), replicates


def cluster_paired_analysis(
    rows: Iterable[Mapping[str, object]],
    *,
    metric: str,
    treatment: str,
    control: str,
    cluster_key: str = "cluster_id",
    condition_key: str = "configuration",
    bootstrap_replicates: int = 10000,
    permutation_replicates: int = 50000,
    seed: int = 20260830,
) -> ClusterPairedResult:
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    grouped: dict[tuple[str, str], list[float]] = {}
    all_clusters: set[str] = set()
    for row in rows:
        missing = [key for key in (cluster_key, condition_key, metric) if key not in row]
        if missing:
            raise ValueError(f"row is missing required fields: {', '.join(missing)}")
        cluster = str(row[cluster_key]).strip()
        condition = str(row[condition_key]).strip()
        if not cluster or not condition:
            raise ValueError("cluster and configuration values must not be blank")
        try:
            value = float(row[metric])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metric {metric!r} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"metric {metric!r} contains a non-finite value")
        grouped.setdefault((cluster, condition), []).append(value)
        all_clusters.add(cluster)

    paired_clusters = sorted(
        cluster
        for cluster in all_clusters
        if (cluster, treatment) in grouped and (cluster, control) in grouped
    )
    if len(paired_clusters) < 2:
        raise ValueError("at least two clusters with both conditions are required")

    treatment_values = [mean(grouped[(cluster, treatment)]) for cluster in paired_clusters]
    control_values = [mean(grouped[(cluster, control)]) for cluster in paired_clusters]
    differences = [a - b for a, b in zip(treatment_values, control_values)]
    rng = random.Random(seed)
    bootstrap = [
        mean([differences[rng.randrange(len(differences))] for _ in differences])
        for _ in range(bootstrap_replicates)
    ]
    p_value, used_permutations = _sign_flip_p_value(
        differences, rng=rng, replicates=permutation_replicates
    )
    diff_sd = stdev(differences) if len(differences) > 1 else 0.0
    control_mean = mean(control_values)
    difference_mean = mean(differences)
    return ClusterPairedResult(
        metric=metric,
        treatment=treatment,
        control=control,
        clusters_used=len(paired_clusters),
        clusters_dropped=len(all_clusters) - len(paired_clusters),
        treatment_mean=mean(treatment_values),
        control_mean=control_mean,
        mean_difference=difference_mean,
        percent_change=(100.0 * difference_mean / control_mean) if control_mean else None,
        ci_low=_quantile(bootstrap, 0.025),
        ci_high=_quantile(bootstrap, 0.975),
        paired_standardized_effect=(difference_mean / diff_sd) if diff_sd else None,
        sign_flip_p_value=p_value,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=used_permutations,
        seed=seed,
    )


def cluster_independent_analysis(
    rows: Iterable[Mapping[str, object]],
    *,
    metric: str,
    treatment: str,
    control: str,
    cluster_key: str = "session_id",
    condition_key: str = "system",
    bootstrap_replicates: int = 10000,
    permutation_replicates: int = 50000,
    seed: int = 20260830,
) -> ClusterIndependentResult:
    """Compare two independent conditions after aggregating within clusters."""
    if bootstrap_replicates < 100 or permutation_replicates < 100:
        raise ValueError("bootstrap and permutation replicates must each be at least 100")
    grouped: dict[tuple[str, str], list[float]] = {}
    membership: dict[str, set[str]] = {}
    for row in rows:
        missing = [key for key in (cluster_key, condition_key, metric) if key not in row]
        if missing:
            raise ValueError(f"row is missing required fields: {', '.join(missing)}")
        cluster = str(row[cluster_key]).strip()
        condition = str(row[condition_key]).strip()
        if not cluster or not condition:
            raise ValueError("cluster and condition values must not be blank")
        if condition not in {treatment, control}:
            continue
        try:
            value = float(row[metric])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metric {metric!r} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"metric {metric!r} contains a non-finite value")
        grouped.setdefault((cluster, condition), []).append(value)
        membership.setdefault(cluster, set()).add(condition)

    overlapping = sorted(cluster for cluster, groups in membership.items() if len(groups) > 1)
    if overlapping:
        raise ValueError(
            "cluster IDs occur in both conditions; use cluster_paired_analysis for paired data"
        )
    treatment_values = [
        mean(values) for (cluster, condition), values in grouped.items() if condition == treatment
    ]
    control_values = [
        mean(values) for (cluster, condition), values in grouped.items() if condition == control
    ]
    if len(treatment_values) < 2 or len(control_values) < 2:
        raise ValueError("at least two independent clusters per condition are required")

    rng = random.Random(seed)
    bootstrap = []
    for _ in range(bootstrap_replicates):
        sampled_treatment = [rng.choice(treatment_values) for _ in treatment_values]
        sampled_control = [rng.choice(control_values) for _ in control_values]
        bootstrap.append(mean(sampled_treatment) - mean(sampled_control))

    observed = abs(mean(treatment_values) - mean(control_values))
    combined = treatment_values + control_values
    treatment_n = len(treatment_values)
    exceedances = 0
    for _ in range(permutation_replicates):
        permuted = combined[:]
        rng.shuffle(permuted)
        difference = mean(permuted[:treatment_n]) - mean(permuted[treatment_n:])
        exceedances += abs(difference) >= observed - 1e-12
    p_value = (exceedances + 1.0) / (permutation_replicates + 1.0)

    treatment_mean = mean(treatment_values)
    control_mean = mean(control_values)
    difference_mean = treatment_mean - control_mean
    pooled_numerator = (
        (len(treatment_values) - 1) * (stdev(treatment_values) ** 2)
        + (len(control_values) - 1) * (stdev(control_values) ** 2)
    )
    pooled_denominator = len(treatment_values) + len(control_values) - 2
    pooled_sd = math.sqrt(pooled_numerator / pooled_denominator) if pooled_denominator else 0.0
    total_n = len(treatment_values) + len(control_values)
    correction = 1.0 - 3.0 / (4.0 * total_n - 9.0)
    hedges_g = correction * difference_mean / pooled_sd if pooled_sd else None
    return ClusterIndependentResult(
        metric=metric,
        treatment=treatment,
        control=control,
        treatment_clusters=len(treatment_values),
        control_clusters=len(control_values),
        treatment_mean=treatment_mean,
        control_mean=control_mean,
        mean_difference=difference_mean,
        percent_change=(100.0 * difference_mean / control_mean) if control_mean else None,
        ci_low=_quantile(bootstrap, 0.025),
        ci_high=_quantile(bootstrap, 0.975),
        hedges_g=hedges_g,
        permutation_p_value=p_value,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        seed=seed,
    )
