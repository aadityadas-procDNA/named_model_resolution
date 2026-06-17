"""
select_measure_column — shared fallback for all model runners.

Priority:
  1. First column with semantic_subtype == "measure"  →  no warning
  2. Highest-scoring "unclassified_metric" column     →  warning emitted
  3. No candidates                                     →  (None, None)

Scoring (unclassified_metric candidates only):
  business_hint present           +2.0   (user explicitly annotated this column)
  null_pct                        -3.0 × null_pct   (penalise missing data)
  CV = std / |mean|  (capped 2.0) +0.5 × min(cv, 2.0)  (reward variation)
  unique_count > 50               +0.5
  unique_count > 10               +0.2
  skewness in (0, 5)              +0.2   (typical for positive business metrics)
  value_max > 0                   +0.1
"""

from __future__ import annotations

from named_model_resolution.models import ColumnProfile, ColumnSpec


def select_measure_column(
    specs: list[ColumnSpec],
    profiles: dict[str, ColumnProfile],
) -> tuple[str | None, str | None]:
    """
    Return (column_name, warning_or_None).

    If a proper 'measure' column exists, return it with no warning.
    Otherwise score all 'unclassified_metric' candidates and return the best
    with a warning explaining the selection and scores.
    """
    # 1. Proper measure column — first wins, no warning
    for s in specs:
        if s.semantic_subtype == "measure":
            return s.name, None

    # 2. Score unclassified candidates
    candidates = [s for s in specs if s.semantic_subtype == "unclassified_metric"]
    if not candidates:
        return None, None

    def _score(spec: ColumnSpec) -> float:
        score = 0.0
        if spec.business_hint:
            score += 2.0
        p = profiles.get(spec.name)
        if p is None:
            return score
        score -= p.null_pct * 3.0
        if p.std is not None and p.mean is not None and abs(p.mean) > 1e-6:
            score += min(p.std / abs(p.mean), 2.0) * 0.5
        if p.unique_count > 50:
            score += 0.5
        elif p.unique_count > 10:
            score += 0.2
        if p.skewness is not None and 0 < p.skewness < 5:
            score += 0.2
        if p.value_max is not None and p.value_max > 0:
            score += 0.1
        return score

    best = max(candidates, key=_score)
    scores = {s.name: round(_score(s), 3) for s in candidates}
    hint_note = f" (business hint: \"{best.business_hint}\")" if best.business_hint else ""
    warning = (
        f"No 'measure' column found. Selected '{best.name}' from unclassified_metric "
        f"candidates via statistical scoring{hint_note}. "
        f"Scores: {scores}. "
        f"Add '{best.name}' to measure_candidates in candidates.yaml to suppress this warning."
    )
    return best.name, warning
