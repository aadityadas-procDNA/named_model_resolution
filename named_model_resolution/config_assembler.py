"""
Layer 3 — Config assembler.

Turns TableClassification objects (one per dataset in a run) into ranked
ModelConfig lists by scoring each model's routing rule against the columns
present in the fact table.

Scoring formula:
  score = (required subtypes ALL satisfied → 1.0 per required, else 0 total)
        + (optional subtypes present × 0.3)
        + (preferred_measures matched × 0.5)

A model is only a candidate when ALL required subtypes are satisfied.

Returns ModelConfigs ranked descending by score.  Callers decide how many
to use (e.g. top-1 for hard routing, top-N for multi-model evaluation).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import DatamartCatalog, ModelConfig, TableClassification


def _subtypes_present(classification: TableClassification) -> set[str]:
    subtypes: set[str] = set()
    for c in classification.columns:
        subtypes.add(c.semantic_subtype)
        subtypes.update(c.secondary_subtypes)
    return subtypes


def _measure_names_present(classification: TableClassification) -> set[str]:
    """Return normalised column names for measure/unclassified_metric columns."""
    names: set[str] = set()
    for c in classification.columns:
        if c.semantic_subtype in {"measure", "unclassified_metric"}:
            names.add(c.name.lower())
            if c.expanded_name:
                names.add(c.expanded_name.lower())
    return names


def _infer_join_keys(
    fact: TableClassification,
    dim: TableClassification,
) -> dict[str, str]:
    """
    Return {fact_col: dim_col} for key columns shared between tables.

    Two-pass matching (gold layer often has no formal DDL schema):
      1. Exact lowercase name match — original behaviour.
      2. Suffix-stripped match: strip _id / _sk / _npi / _code / _key from
         both names and retry.  Handles NPI_ID <-> NPI, TERR_ID <-> TERRITORY_SK, etc.
    """
    _STRIP = ("_sk", "_id", "_npi", "_code", "_key", "_vid")

    def _strip_suffix(name: str) -> str:
        for sfx in _STRIP:
            if name.endswith(sfx):
                return name[: -len(sfx)]
        return name

    fact_keys = {c.name.lower(): c.name for c in fact.columns if c.semantic_subtype == "key"}
    dim_keys  = {c.name.lower(): c.name for c in dim.columns  if c.semantic_subtype == "key"}

    # Pass 1: exact match
    result = {fact_keys[k]: dim_keys[k] for k in set(fact_keys) & set(dim_keys)}
    if result:
        return result

    # Pass 2: suffix-stripped match
    fact_stripped = {_strip_suffix(k): orig_lower for k, orig_lower in fact_keys.items()}
    dim_stripped  = {_strip_suffix(k): orig_lower for k, orig_lower in dim_keys.items()}

    for stripped in set(fact_stripped) & set(dim_stripped):
        if not stripped:
            continue
        f_lower = fact_stripped[stripped]
        d_lower = dim_stripped[stripped]
        result[fact_keys[f_lower]] = dim_keys[d_lower]

    return result


def assemble(
    target: TableClassification,
    all_classifications: list[TableClassification],
    catalog: DatamartCatalog,
    configs_dir: str | Path,
) -> list[ModelConfig]:
    """
    Build ranked ModelConfigs for `target` (must be a fact table).
    `all_classifications` is the full list so we can find joinable dimensions.
    """
    configs_dir = Path(configs_dir)
    with (configs_dir / "model_routing.yaml").open() as f:
        routing_rules: dict = yaml.safe_load(f) or {}

    if target.table_type != "fact":
        return []

    # ── Star schema: 1-hop dimension join detection ───────────────────────────
    # Find all dimensions with at least one matching key column to this fact table.
    # Their subtypes are inherited into the effective subtype set for routing.
    # This is intentionally 1-hop only (no dim→dim traversal = no snowflake cost).
    dim_tables = [tc for tc in all_classifications if tc.table_type == "dimension"]

    join_keys: dict[str, str] = {}
    matched_dims: list[str] = []
    for dim in dim_tables:
        keys = _infer_join_keys(target, dim)
        if keys:
            join_keys.update(keys)
            matched_dims.append(dim.table_name)

    # Effective subtypes = fact's own + inherited from joined dims (1-hop)
    subtypes = _subtypes_present(target)
    for dim in dim_tables:
        if dim.table_name in matched_dims:
            # Inherit routing-relevant subtypes only (skip unknown/unmatched)
            inherited = _subtypes_present(dim) - {"unknown", "unmatched"}
            subtypes = subtypes | inherited

    measure_names = _measure_names_present(target)
    unclassified_cols = [
        c.name for c in target.columns if c.semantic_subtype == "unclassified_metric"
    ]

    # Derive use-case hints from matched catalog entry
    catalog_use_cases: list[str] = []
    if target.matched_catalog_entry and target.matched_catalog_entry in catalog.datamarts:
        desc = catalog.datamarts[target.matched_catalog_entry].description
        if desc:
            catalog_use_cases.append(desc)

    results: list[ModelConfig] = []

    for model_name, rule in routing_rules.items():
        required: list[str] = rule.get("required", [])
        optional: list[str] = rule.get("optional", [])
        accepts: list[str] = rule.get("accepts", [])
        preferred_measures: list[str] = rule.get("preferred_measures", [])
        rule_use_cases: list[str] = rule.get("use_case_hints", [])

        # Gate: all required subtypes must be present (fact + inherited dim subtypes)
        if not all(r in subtypes for r in required):
            continue

        # Score
        score = float(len(required))  # 1.0 per required (all satisfied at this point)
        score += sum(0.3 for opt in optional if opt in subtypes)
        score += sum(0.5 for pm in preferred_measures if pm in measure_names)

        # Normalise by theoretical max to get a 0-1 confidence
        max_score = len(required) + len(optional) * 0.3 + len(preferred_measures) * 0.5
        confidence = round(score / max_score, 3) if max_score > 0 else 0.0

        # Pass-through unclassified metrics if model accepts them
        flagged = unclassified_cols if "unclassified_metric" in accepts else []

        use_cases = catalog_use_cases + rule_use_cases

        results.append(
            ModelConfig(
                model_name=model_name,
                confidence=confidence,
                fact_table=target.table_name,
                dimension_tables=matched_dims,
                join_keys=join_keys,
                use_cases=use_cases,
                flagged_unclassified_columns=flagged,
            )
        )

    results.sort(key=lambda m: m.confidence, reverse=True)
    return results
