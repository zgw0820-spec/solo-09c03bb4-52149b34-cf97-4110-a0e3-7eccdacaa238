"""Normalisation of user-supplied training input.

The API accepts raw fan vibration/current features and standardises each
column with statistics from the submitted sample (zero mean, unit variance);
constant columns become zero.  A bias column of ones is appended, so the
learned vector is ``[w_1 .. w_d, b]``.  Labels must be 0 (healthy) or 1
(shutdown warning).

The raw per-column statistics (mean, standard deviation and which columns are
constant) are returned alongside the design matrix.  They are persisted for
every job so a continuation job can map weights trained on one sample to the
standardisation scale of another sample without trying to invert the
standardised matrix.

The float64 design matrix is persisted, so any worker resumes on bit-identical
inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_RECORDS = 8
MAX_RECORDS = 64
MIN_EPOCHS = 2
MAX_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.1
MOMENTUM = 0.9
# A column with a smaller population standard deviation is treated as constant.
CONSTANT_EPS = 1e-12


class InputError(ValueError):
    """Invalid training submission."""


@dataclass(frozen=True)
class FeatureStats:
    """Raw-column statistics used to standardise one submitted sample."""

    mean: np.ndarray
    std: np.ndarray
    constant: np.ndarray  # boolean mask, one entry per raw feature column


def _validate(
    features: list[list[float]], labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    if not (MIN_RECORDS <= len(features) <= MAX_RECORDS):
        raise InputError(f"submit between {MIN_RECORDS} and {MAX_RECORDS} records")
    if len(labels) != len(features):
        raise InputError("features and labels must contain the same number of records")
    if any(len(row) == 0 for row in features):
        raise InputError("each record needs at least one feature")
    if any(len(row) != len(features[0]) for row in features):
        raise InputError("every record must have the same number of features")

    try:
        matrix = np.asarray(features, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError("features and labels must be numeric") from exc
    if matrix.ndim != 2:
        raise InputError("features must be a two-dimensional list of numbers")
    if not np.all(np.isfinite(matrix)):
        raise InputError("features must be finite numbers")
    if not np.all((targets == 0.0) | (targets == 1.0)):
        raise InputError("labels must be 0 or 1")
    return matrix, targets


def normalise_samples(
    features: list[list[float]], labels: list[int]
) -> tuple[np.ndarray, np.ndarray, FeatureStats]:
    """Validate input and return ``(design, targets, raw_stats)``."""
    matrix, targets = _validate(features, labels)

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    constant = std < CONSTANT_EPS
    scaled = (matrix - mean) / np.where(constant, 1.0, std)
    scaled[:, constant] = 0.0
    design = np.concatenate([scaled, np.ones((scaled.shape[0], 1), dtype=np.float64)], axis=1)
    return design, targets, FeatureStats(mean=mean, std=std, constant=constant)


def normalise(features: list[list[float]], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    design, targets, _ = normalise_samples(features, labels)
    return design, targets


def rescale_weights(
    source_weights: np.ndarray,
    source_stats: FeatureStats,
    target_stats: FeatureStats,
) -> np.ndarray:
    """Re-express ``source_weights`` on the target sample's standardisation.

    A source model scores a raw record ``x`` as::

        logit = sum_i a_i * (x_i - ms_i) / ss_i + b

    after the target sample is standardised with ``(mt, st)`` the worker will
    compute ``sum_i c_i * (x_i - mt_i) / st_i + d``.  Choosing
    ``c_i = a_i * st_i / ss_i`` and

        d = b + sum_i a_i * (mt_i - ms_i) / ss_i

    makes the raw score identical, so the continuation starts exactly where the
    source model left off.

    Constant source columns have ``a_i == 0`` (their standardised design column
    is always zero, so training never moves that weight) and their term is zero.
    A constant *target* column has no scale: every record in the new sample
    carries the same value, so its whole source contribution is a constant and
    is folded into ``d``.  Its new weight stays zero, because that standardised
    design column is zero too.  Predictions on the new sample's raw
    measurements therefore still match the source model exactly.
    """
    source = np.asarray(source_weights, dtype=np.float64)
    mean_s = np.asarray(source_stats.mean, dtype=np.float64)
    std_s = np.asarray(source_stats.std, dtype=np.float64)
    constant_s = np.asarray(source_stats.constant, dtype=bool)
    mean_t = np.asarray(target_stats.mean, dtype=np.float64)
    std_t = np.asarray(target_stats.std, dtype=np.float64)
    constant_t = np.asarray(target_stats.constant, dtype=bool)

    dim = mean_s.shape[0]
    if mean_t.shape[0] != dim or std_t.shape[0] != dim or constant_t.shape[0] != dim:
        raise InputError("source and new sample must have the same feature columns")
    if source.shape[0] != dim + 1:
        raise InputError("source weights do not match the source feature columns")

    if np.any(constant_s & (source[:dim] != 0.0)):
        raise InputError("source weights on a constant source column are inconsistent")
    if not np.all(np.isfinite(source)):
        raise InputError("source weights must be finite")
    if not np.all(np.isfinite(mean_t)) or not np.all(np.isfinite(mean_s)):
        raise InputError("column statistics must be finite")
    if not np.all(std_t[~constant_t] > 0.0) or not np.all(std_s[~constant_s] > 0.0):
        raise InputError("column statistics must be non-degenerate")

    scale = np.zeros(dim, dtype=np.float64)
    usable = ~constant_t
    scale[usable] = std_t[usable] / np.where(constant_s, 1.0, std_s)[usable]
    scaled = source[:dim] * scale

    # The raw-scale intercept shift.  For a constant source column a_i == 0, so
    # using 1 in the denominator is exact, and a constant target column is
    # folded in here (its column cannot carry the term itself).
    source_scale = np.where(constant_s, 1.0, std_s)
    bias = float(source[dim] + np.sum(source[:dim] * (mean_t - mean_s) / source_scale))
    weights = np.concatenate([scaled, [bias]])
    if not np.all(np.isfinite(weights)):
        raise InputError("rescaled weights are not finite")
    return weights
