"""
Quality gate check functions -- pure, stateless.

Each function signature:
    check_fn(
        df: pd.DataFrame,
        column_specs: list[ColumnSpec],
        column_profiles: list[ColumnProfile],
        params: dict,            # merged global + model thresholds
    ) -> QualityCheckResult

Reuses pre-computed stats from ColumnProfile wherever possible
(null_pct, skewness, unique_count) so no redundant sampling.
Only date_continuity, channel_collinearity, and autocorrelation
need the actual df.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from named_model_resolution.models import ColumnProfile, ColumnSpec

from .models import QualityCheckResult


# -- Helpers ------------------------------------------------------------------

def _profile_map(column_profiles: list[ColumnProfile]) -> dict[str, ColumnProfile]:
    return {p.name: p for p in column_profiles}


def _specs_by_subtype(
    column_specs: list[ColumnSpec],
    *subtypes: str,
) -> list[ColumnSpec]:
    return [s for s in column_specs if s.semantic_subtype in subtypes]


# -- Check functions ----------------------------------------------------------

def fill_rate(
    df: pd.DataFrame,
    column_specs: list[ColumnSpec],
    column_profiles: list[ColumnProfile],
    params: dict,
) -> QualityCheckResult:
    """
    Check null percentage of key columns (date, measure, channel).
    Uses pre-computed ColumnProfile.null_pct -- no re-sample needed.
    """
    min_fill = params.get("min_fill_rate", 0.80)
    fail_fill = params.get("fail_fill_rate", 0.50)

    key_specs = _specs_by_subtype(column_specs, "date", "measure", "channel")
    if not key_specs:
        return QualityCheckResult(
            check_name="fill_rate",
            status="WARN",
            detail="no key columns (date/measure/channel) identified for fill-rate check",
            metric=None,
        )

    profiles = _profile_map(column_profiles)
    worst_col = None
    worst_fill = 1.0

    for s in key_specs:
        p = profiles.get(s.name)
        if p is None:
            continue
        fill = 1.0 - p.null_pct
        if fill < worst_fill:
            worst_fill = fill
            worst_col = s.name

    if worst_col is None:
        return QualityCheckResult(
            check_name="fill_rate",
            status="PASS",
            detail="no profiled key columns to check",
            metric=None,
        )

    if worst_fill < fail_fill:
        return QualityCheckResult(
            check_name="fill_rate",
            status="FAIL",
            detail=f"column '{worst_col}' fill rate {worst_fill:.0%} < {fail_fill:.0%} (unusable)",
            metric=round(worst_fill, 4),
        )
    if worst_fill < min_fill:
        return QualityCheckResult(
            check_name="fill_rate",
            status="WARN",
            detail=f"column '{worst_col}' fill rate {worst_fill:.0%} < {min_fill:.0%} threshold",
            metric=round(worst_fill, 4),
        )
    return QualityCheckResult(
        check_name="fill_rate",
        status="PASS",
        detail=f"all key columns >={worst_fill:.0%} filled (worst: '{worst_col}')",
        metric=round(worst_fill, 4),
    )


def zero_variance(
    df: pd.DataFrame,
    column_specs: list[ColumnSpec],
    column_profiles: list[ColumnProfile],
    params: dict,
) -> QualityCheckResult:
    """
    Check coefficient of variation (CV = std/|mean|) for measure/channel columns.
    Computes CV directly from df (ColumnProfile does not store std/mean).
    """
    cv_threshold = params.get("zero_variance_cv", 0.01)

    target_specs = _specs_by_subtype(column_specs, "measure", "channel", "unclassified_metric")
    if not target_specs:
        return QualityCheckResult(
            check_name="zero_variance",
            status="PASS",
            detail="no measure/channel columns to check",
            metric=None,
        )

    near_constant = []
    for s in target_specs:
        if s.name not in df.columns:
            continue
        col_data = pd.to_numeric(df[s.name], errors="coerce").dropna()
        if len(col_data) < 2:
            continue
        mean_val = col_data.mean()
        std_val = col_data.std()
        cv = std_val / (abs(mean_val) + 1e-12)
        if cv < cv_threshold:
            near_constant.append((s.name, round(cv, 6)))

    if near_constant:
        names = ", ".join(f"'{n}' (CV={v})" for n, v in near_constant[:5])
        return QualityCheckResult(
            check_name="zero_variance",
            status="WARN",
            detail=f"near-constant columns (CV<{cv_threshold}): {names}",
            metric=near_constant[0][1],
        )
    return QualityCheckResult(
        check_name="zero_variance",
        status="PASS",
        detail=f"all measure/channel columns have CV >= {cv_threshold}",
        metric=None,
    )


def date_continuity(
    df: pd.DataFrame,
    column_specs: list[ColumnSpec],
    column_profiles: list[ColumnProfile],
    params: dict,
) -> QualityCheckResult:
    """
    Check for gaps in the date column.
    Needs the actual df to compute consecutive diffs.
    """
    max_gap_weeks = params.get("max_gap_weeks", 4)
    min_weeks = params.get("min_weeks", params.get("min_row_count", 52))

    date_specs = _specs_by_subtype(column_specs, "date")
    if not date_specs:
        return QualityCheckResult(
            check_name="date_continuity",
            status="WARN",
            detail="no date column identified",
            metric=None,
        )

    date_col = date_specs[0].name
    if date_col not in df.columns:
        return QualityCheckResult(
            check_name="date_continuity",
            status="WARN",
            detail=f"date column '{date_col}' not in dataframe",
            metric=None,
        )

    try:
        dates = pd.to_datetime(df[date_col]).dropna().sort_values().unique()
        n_weeks = len(dates)

        if n_weeks < min_weeks:
            return QualityCheckResult(
                check_name="date_continuity",
                status="FAIL",
                detail=f"only {n_weeks} unique dates -- need >= {min_weeks}",
                metric=float(n_weeks),
            )

        if n_weeks < 2:
            return QualityCheckResult(
                check_name="date_continuity",
                status="PASS",
                detail="single date point -- cannot check continuity",
                metric=float(n_weeks),
            )

        gaps_days = pd.Series(dates).diff().dropna().dt.days
        max_gap_days = int(gaps_days.max())
        max_gap_w = max_gap_days / 7.0
        n_gaps = int((gaps_days > max_gap_weeks * 7).sum())

        if max_gap_w > max_gap_weeks:
            return QualityCheckResult(
                check_name="date_continuity",
                status="WARN",
                detail=f"{n_gaps} gap(s) > {max_gap_weeks} weeks detected "
                       f"(largest ~= {max_gap_w:.1f} weeks)",
                metric=round(max_gap_w, 2),
            )
        return QualityCheckResult(
            check_name="date_continuity",
            status="PASS",
            detail=f"no gaps > {max_gap_weeks} weeks in {n_weeks}-week series",
            metric=0.0,
        )
    except Exception as exc:
        return QualityCheckResult(
            check_name="date_continuity",
            status="WARN",
            detail=f"date continuity check failed: {exc}",
            metric=None,
        )


def channel_collinearity(
    df: pd.DataFrame,
    column_specs: list[ColumnSpec],
    column_profiles: list[ColumnProfile],
    params: dict,
) -> QualityCheckResult:
    """
    Pairwise Pearson correlation between channel columns.
    High collinearity makes MMM coefficients unstable.
    """
    threshold = params.get("collinearity_threshold", 0.85)
    min_channel_count = params.get("min_channel_count", 2)

    ch_specs = _specs_by_subtype(column_specs, "channel")
    ch_cols = [s.name for s in ch_specs if s.name in df.columns]

    if len(ch_cols) < min_channel_count:
        return QualityCheckResult(
            check_name="channel_collinearity",
            status="FAIL",
            detail=f"only {len(ch_cols)} channel column(s) found "
                   f"-- need >= {min_channel_count} for MMM",
            metric=float(len(ch_cols)),
        )

    if len(ch_cols) < 2:
        return QualityCheckResult(
            check_name="channel_collinearity",
            status="PASS",
            detail="single channel -- no collinearity to check",
            metric=None,
        )

    try:
        ch_data = df[ch_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(ch_data) < 3:
            return QualityCheckResult(
                check_name="channel_collinearity",
                status="WARN",
                detail="too few rows to compute reliable correlations",
                metric=None,
            )

        corr = ch_data.corr().values
        n = len(ch_cols)
        hi_pairs = [
            (ch_cols[i], ch_cols[j], float(corr[i, j]))
            for i in range(n) for j in range(i + 1, n)
            if abs(corr[i, j]) > threshold
        ]
        max_r = max((abs(corr[i, j]) for i in range(n) for j in range(i + 1, n)),
                    default=0.0)

        if hi_pairs:
            pair_str = "; ".join(
                f"'{a}' <-> '{b}' r={r:.2f}" for a, b, r in hi_pairs[:3]
            )
            return QualityCheckResult(
                check_name="channel_collinearity",
                status="WARN",
                detail=f"{len(hi_pairs)} high-collinearity pair(s) (|r|>{threshold}): {pair_str}",
                metric=round(max_r, 4),
            )
        return QualityCheckResult(
            check_name="channel_collinearity",
            status="PASS",
            detail=f"no channel pair has |r| > {threshold} (max observed: {max_r:.3f})",
            metric=round(max_r, 4),
        )
    except Exception as exc:
        return QualityCheckResult(
            check_name="channel_collinearity",
            status="WARN",
            detail=f"collinearity check failed: {exc}",
            metric=None,
        )


def segment_balance(
    df: pd.DataFrame,
    column_specs: list[ColumnSpec],
    column_profiles: list[ColumnProfile],
    params: dict,
) -> QualityCheckResult:
    """
    Check that segment groups are not severely imbalanced.
    Uses ColumnProfile.unique_count for a quick guard + df for actual counts.
    """
    max_imbalance = params.get("max_imbalance_ratio", 10.0)
    min_seg_size = params.get("min_segment_size", 30)

    seg_specs = _specs_by_subtype(column_specs, "segment")
    if not seg_specs:
        return QualityCheckResult(
            check_name="segment_balance",
            status="WARN",
            detail="no segment column identified",
            metric=None,
        )

    seg_col = seg_specs[0].name
    if seg_col not in df.columns:
        return QualityCheckResult(
            check_name="segment_balance",
            status="WARN",
            detail=f"segment column '{seg_col}' not in dataframe",
            metric=None,
        )

    try:
        counts = df[seg_col].value_counts()
        if len(counts) == 0:
            return QualityCheckResult(
                check_name="segment_balance",
                status="FAIL",
                detail=f"segment column '{seg_col}' has no values",
                metric=None,
            )

        min_count = int(counts.min())
        max_count = int(counts.max())
        ratio = max_count / (min_count + 1e-9)

        if min_count < min_seg_size:
            return QualityCheckResult(
                check_name="segment_balance",
                status="FAIL",
                detail=f"smallest segment in '{seg_col}' has only {min_count} rows "
                       f"(need >= {min_seg_size})",
                metric=float(min_count),
            )
        if ratio > max_imbalance:
            return QualityCheckResult(
                check_name="segment_balance",
                status="WARN",
                detail=f"imbalance ratio {ratio:.1f}x > {max_imbalance}x "
                       f"in '{seg_col}' ({len(counts)} segments)",
                metric=round(ratio, 2),
            )
        return QualityCheckResult(
            check_name="segment_balance",
            status="PASS",
            detail=f"'{seg_col}': {len(counts)} segments, ratio {ratio:.1f}x <= {max_imbalance}x",
            metric=round(ratio, 2),
        )
    except Exception as exc:
        return QualityCheckResult(
            check_name="segment_balance",
            status="WARN",
            detail=f"segment balance check failed: {exc}",
            metric=None,
        )


def autocorrelation(
    df: pd.DataFrame,
    column_specs: list[ColumnSpec],
    column_profiles: list[ColumnProfile],
    params: dict,
) -> QualityCheckResult:
    """
    Lag-1 autocorrelation of the target (measure) series.
    Low ACF means little temporal structure -- ARIMA gains nothing.
    """
    min_acf = params.get("min_acf_lag1", 0.10)

    measure_specs = _specs_by_subtype(column_specs, "measure")
    if not measure_specs:
        return QualityCheckResult(
            check_name="autocorrelation",
            status="WARN",
            detail="no measure column identified for autocorrelation check",
            metric=None,
        )

    measure_col = measure_specs[0].name
    if measure_col not in df.columns:
        return QualityCheckResult(
            check_name="autocorrelation",
            status="WARN",
            detail=f"measure column '{measure_col}' not in dataframe",
            metric=None,
        )

    date_specs = _specs_by_subtype(column_specs, "date")
    try:
        series = pd.to_numeric(df[measure_col], errors="coerce").dropna()
        if date_specs and date_specs[0].name in df.columns:
            date_col = date_specs[0].name
            sorted_df = df[[date_col, measure_col]].copy()
            sorted_df[date_col] = pd.to_datetime(sorted_df[date_col])
            sorted_df = sorted_df.sort_values(date_col)
            series = pd.to_numeric(sorted_df[measure_col], errors="coerce").dropna()

        if len(series) < 10:
            return QualityCheckResult(
                check_name="autocorrelation",
                status="WARN",
                detail="too few observations to compute reliable ACF",
                metric=None,
            )

        acf_lag1 = float(series.autocorr(lag=1))

        if abs(acf_lag1) < min_acf:
            return QualityCheckResult(
                check_name="autocorrelation",
                status="WARN",
                detail=f"lag-1 ACF={acf_lag1:.3f} < {min_acf} -- "
                       "limited temporal structure for ARIMA",
                metric=round(acf_lag1, 4),
            )
        return QualityCheckResult(
            check_name="autocorrelation",
            status="PASS",
            detail=f"lag-1 ACF={acf_lag1:.3f} >= {min_acf}",
            metric=round(acf_lag1, 4),
        )
    except Exception as exc:
        return QualityCheckResult(
            check_name="autocorrelation",
            status="WARN",
            detail=f"autocorrelation check failed: {exc}",
            metric=None,
        )


def min_row_count(
    df: pd.DataFrame,
    column_specs: list[ColumnSpec],
    column_profiles: list[ColumnProfile],
    params: dict,
) -> QualityCheckResult:
    """Verify the dataset has enough rows for the target model."""
    # Model-specific minimum supersedes global minimum
    required = params.get("min_rows", params.get("min_weeks", params.get("min_row_count", 52)))
    n = len(df)

    if n < required:
        return QualityCheckResult(
            check_name="min_row_count",
            status="FAIL",
            detail=f"only {n} rows -- need >= {required}",
            metric=float(n),
        )
    return QualityCheckResult(
        check_name="min_row_count",
        status="PASS",
        detail=f"{n} rows >= {required} required",
        metric=float(n),
    )
