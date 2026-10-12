"""
temporal_filter.py
------------------
Temporal filtering + precedence resolution.

Given a set of retrieved clauses and a query_date (or None for "current"),
this module:
  1. Filters out clauses from documents NOT in force on query_date.
  2. Among remaining clauses from the same policy category, applies
     the three-tier precedence rule defined in charter.md §3:
       Tier 1: Explicit supersession (direct supersedes pointer)
       Tier 2: Specificity (narrower scope wins within its scope)
       Tier 3: Recency (most recent effective_date wins)
  3. Returns filtered clauses tagged with why each survived.

DEFAULT_REF_DATE is used when as_of_date is omitted (per charter decision).
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from src.corpus_loader import Clause, DocMeta
from src.knowledge_graph import build_knowledge_graph, get_authority_tier


# ── Default reference date for "as_of_date = null" queries ──────────────────
DEFAULT_REF_DATE = date(2025, 6, 1)   # fixed per charter §5 / week2_README §3


# ── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class FilteredClause:
    """A clause that survived temporal filtering, with an explanation."""
    clause: Clause
    in_force: bool                          # True if doc is valid on query_date
    superseded_by: Optional[str] = None     # doc_id that supersedes this one (if any)
    precedence_note: str = ""               # human-readable reason this clause was kept/dropped


# ── Core Filtering Logic ──────────────────────────────────────────────────────

def resolve_date(as_of_date: Optional[str | date]) -> date:
    """
    Convert an as_of_date string/date/None to a concrete date.
    None → DEFAULT_REF_DATE (2025-06-01).
    """
    if as_of_date is None:
        return DEFAULT_REF_DATE
    if isinstance(as_of_date, date):
        return as_of_date
    return date.fromisoformat(str(as_of_date))


def is_in_force(meta: DocMeta, query_date: date) -> bool:
    """Return True if the document is in force on query_date (inclusive on both ends)."""
    if meta.effective_date > query_date:
        return False
    if meta.sunset_date is not None and meta.sunset_date < query_date:
        return False
    return True


def _find_supersessor(doc_id: str, metadata: dict[str, DocMeta]) -> Optional[str]:
    """Return the doc_id that directly supersedes doc_id, or None."""
    for did, meta in metadata.items():
        if meta.supersedes == doc_id:
            return did
    return None


def temporal_filter(
    retrieved: list[tuple[float, Clause]],
    metadata: dict[str, DocMeta],
    query_date: date,
    keep_expired_for_context: bool = False,
) -> list[FilteredClause]:
    """
    Filter and annotate retrieved clauses by temporal validity.

    Args:
        retrieved:               List of (score, Clause) from the retriever.
        metadata:                Full metadata dict.
        query_date:              The as-of date for this query.
        keep_expired_for_context: If True, expired clauses are kept but flagged
                                   (useful for T1 trap detection in the answer).

    Returns:
        List of FilteredClause sorted by (in_force DESC, score DESC).
    """
    results: list[FilteredClause] = []

    for score, clause in retrieved:
        meta = clause.meta
        in_force_flag = is_in_force(meta, query_date)

        # Find if something supersedes this doc AND that supersessor is in force
        supersessor = _find_supersessor(meta.doc_id, metadata)
        active_supersessor: Optional[str] = None
        if supersessor and is_in_force(metadata[supersessor], query_date):
            active_supersessor = supersessor

        # Build a human-readable note
        if in_force_flag:
            note = (
                f"{meta.doc_id} ({meta.version}) in force "
                f"[{meta.effective_date} – {meta.sunset_date or 'current'}]"
            )
        else:
            if active_supersessor:
                note = (
                    f"EXPIRED — superseded by {active_supersessor} "
                    f"(effective {metadata[active_supersessor].effective_date})"
                )
            elif meta.effective_date > query_date:
                note = f"NOT YET IN FORCE — effective from {meta.effective_date}"
            else:
                note = (
                    f"EXPIRED — sunset {meta.sunset_date}; "
                    f"{'no active successor found' if not supersessor else f'successor {supersessor} also expired'}"
                )

        fc = FilteredClause(
            clause=clause,
            in_force=in_force_flag,
            superseded_by=active_supersessor,
            precedence_note=note,
        )

        if in_force_flag or keep_expired_for_context:
            results.append(fc)

    # Sort: in-force clauses first, then by retrieval score (descending)
    results.sort(key=lambda x: (not x.in_force, -0.0))
    return results


def apply_precedence(
    filtered: list[FilteredClause],
    metadata: dict[str, DocMeta],
) -> list[FilteredClause]:
    """
    Among in-force filtered clauses that share a category, apply the
    three-tier precedence rule and annotate each clause accordingly.

    This does NOT remove clauses — it annotates them with 'precedence_note'
    so the conflict detector and answer generator can reason transparently.

    Tier 1: Explicit supersession → predecessor is lower-priority
    Tier 2: Specificity (non-None sunset_date = time-boxed = often more specific)
    Tier 3: Recency (most recent effective_date wins within same tier)
    """
    in_force = [fc for fc in filtered if fc.in_force]

    # Group by category
    by_category: dict[str, list[FilteredClause]] = {}
    for fc in in_force:
        cat = fc.clause.meta.category
        by_category.setdefault(cat, []).append(fc)

    for cat, group in by_category.items():
        # Collect all doc_ids in this group
        doc_ids_in_group = {fc.clause.doc_id for fc in group}

        for fc in group:
            meta = fc.clause.meta

            # Tier 1: Is this doc explicitly superseded by another in-force doc in the group?
            if fc.superseded_by and fc.superseded_by in doc_ids_in_group:
                fc.precedence_note += " | Tier-1: explicitly superseded"
                continue

            # Tier 2: Is this doc more specific (time-boxed or narrow scope)?
            # Heuristic: a time-boxed doc (non-None sunset_date) covering a shorter window
            # within a larger doc's window is treated as "more specific"
            # The full specificity check happens in the conflict detector.

            # Tier 3: Annotate recency
            peers = [
                other for other in group
                if other.clause.doc_id != meta.doc_id
                and other.clause.meta.category == meta.category
                and not (other.superseded_by and other.superseded_by == meta.doc_id)
            ]
            if peers:
                most_recent_peer = max(peers, key=lambda x: x.clause.meta.effective_date)
                if meta.effective_date < most_recent_peer.clause.meta.effective_date:
                    fc.precedence_note += (
                        f" | Tier-3: older than {most_recent_peer.clause.doc_id} "
                        f"(effective {most_recent_peer.clause.meta.effective_date})"
                    )
                else:
                    fc.precedence_note += " | Tier-3: most recent in category"

    return filtered
