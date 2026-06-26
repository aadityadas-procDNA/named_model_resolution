from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnSpec:
    name: str
    dtype: str
    semantic_subtype: str
    # "date" | "geography" | "measure" | "key" | "flag" | "segment" |
    # "dimension_attribute" | "unclassified_metric" | "unknown"
    match_source: str
    # "candidate_list" | "abbreviation_expanded" | "heuristic_token" |
    # "heuristic_dtype" | "guardrail_metric" | "unmatched"
    expanded_name: str | None = None  # e.g. "WK_END" → "week_end_date"
    confidence: float = 0.0
    business_hint: str | None = None  # user-supplied description from candidates.yaml business_hints
    secondary_subtypes: list[str] = field(default_factory=list)
    # Additional valid subtypes beyond the primary.  Valid dual-label pairs are
    # defined by _DUAL_LABEL_PAIRS in column_matcher.py (e.g. channel + measure).


@dataclass
class DatamartSpec:
    name: str
    columns: list[str]
    category: str = ""
    description: str = ""


@dataclass
class DatamartCatalog:
    datamarts: dict[str, DatamartSpec] = field(default_factory=dict)


@dataclass
class TableClassification:
    table_name: str
    table_type: str  # "fact" | "dimension" | "unknown"
    columns: list[ColumnSpec] = field(default_factory=list)
    matched_catalog_entry: str | None = None
    catalog_match_score: float = 0.0


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_pct: float = 0.0
    skewness: float | None = None
    kurtosis: float | None = None
    value_max: float | None = None
    unique_count: int = 0
    date_grain: str | None = None  # "daily" | "weekly" | "monthly" | None
    suggested_transforms: list[str] = field(default_factory=list)
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    outlier_rate: float | None = None  # fraction of values outside 1.5 × IQR fences


@dataclass
class ModelConfig:
    model_name: str
    confidence: float
    fact_table: str
    dimension_tables: list[str] = field(default_factory=list)
    join_keys: dict[str, str] = field(default_factory=dict)  # {fact_col: dim_col}
    use_cases: list[str] = field(default_factory=list)
    flagged_unclassified_columns: list[str] = field(default_factory=list)


@dataclass
class RouterResult:
    dataset_name: str
    classification: TableClassification
    model_configs: list[ModelConfig] = field(default_factory=list)
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Signal deduplication — set by Router after all tables are processed
    is_duplicate_signal: bool = False          # True if another table has richer same-signature routing
    signal_group_primary: str | None = None   # Name of the primary (most informative) table in this group
