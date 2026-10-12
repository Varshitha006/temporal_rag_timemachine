"""
knowledge_graph.py
------------------
Builds a Knowledge Graph (KG) of policy document relationships using NetworkX.

V2: Enhanced with clause-level semantic relationships.

Nodes  = documents (doc_id) — each with metadata attributes
Edges  = relationships between documents:
  - "supersedes"     : version succession chain (D001 → D002 → D003)
  - "same_category"  : documents in the same policy family
  - "supplements"    : same category but different sub-topics (no conflict)
  - "narrows_scope"  : one doc is a scoped specialisation of the other
  - "conflict_risk"  : documents that genuinely conflict (same topic, different rules)

The KG is used by the temporal filter and conflict detector to quickly identify:
  1. Which document is the *active* version for a given date (via supersession chain)
  2. Which documents might *conflict* (same category, overlapping windows,
     AND semantically conflicting clauses — not just same-category)
  3. Which issuing_authority "owns" the policy (for conflict escalation decisions)
"""

from datetime import date
from typing import Optional

import networkx as nx

from src.corpus_loader import DocMeta
from src.clause_relationship import (
    get_doc_pair_relationship,
    is_supplementary,
    is_genuine_conflict,
)


# ── Authority Tier Ranking ────────────────────────────────────────────────
# Used as tie-breaker signal in conflict reporting (NOT auto-ranking).
AUTHORITY_TIERS = {
    "Legal & Compliance": 1,
    "IT Governance": 2,
    "IT Security": 2,
    "Ethics Committee": 2,
    "Finance Department": 3,
    "Procurement Committee": 3,
    "Procurement & Legal": 3,
    "HR & Legal": 3,
    "IT Governance & Legal": 3,
    "Human Resources Department": 4,
    "Corporate Communications & HR": 4,
    "Finance & HR": 4,
    "Executive Office": 4,
    "Bengaluru Regional HR": 5,
    "Field Operations": 5,
    "Sales & Marketing": 5,
    "Regional Sales Leadership (South Region)": 5,
}


def get_authority_tier(authority: str) -> int:
    """Return numeric tier for an issuing authority (1 = highest, higher # = lower)."""
    for key, tier in AUTHORITY_TIERS.items():
        if key.lower() in authority.lower() or authority.lower() in key.lower():
            return tier
    return 6  # unknown/lowest


def _windows_overlap(
    eff1: date, sun1: Optional[date],
    eff2: date, sun2: Optional[date],
) -> bool:
    """Return True if two validity windows [eff1, sun1) and [eff2, sun2) overlap."""
    # Use far-future date as sentinel for open-ended (sun_date = None)
    FAR_FUTURE = date(9999, 12, 31)
    end1 = sun1 if sun1 is not None else FAR_FUTURE
    end2 = sun2 if sun2 is not None else FAR_FUTURE
    # Overlap iff neither one ends before the other starts
    return eff1 <= end2 and eff2 <= end1


# ── Public API ───────────────────────────────────────────────────────────────

def build_knowledge_graph(metadata: dict[str, DocMeta]) -> nx.DiGraph:
    """
    Build and return a directed KG from the metadata dict.

    V2 Enhancement: Uses clause-level semantic relationships to distinguish
    between supplementary, scope-narrowing, and genuinely conflicting document
    pairs. Only genuine conflicts get the "conflict_risk" edge.

    Node attributes stored per doc_id:
      title, issuing_authority, effective_date, sunset_date, version,
      category, authority_tier

    Edge types (stored as edge attribute 'rel'):
      - 'supersedes'    : B supersedes A  →  edge A → B
      - 'same_category' : A and B are in the same policy family
      - 'supplements'   : A and B are same category but cover different sub-topics
      - 'narrows_scope' : A is a scoped specialisation of B (or vice versa)
      - 'conflict_risk' : A and B genuinely conflict (same topic, different rules)
    """
    G = nx.DiGraph()

    # ── Step 1: Add all document nodes with their metadata attributes ────
    for doc_id, meta in metadata.items():
        G.add_node(
            doc_id,
            title=meta.title,
            issuing_authority=meta.issuing_authority,
            effective_date=meta.effective_date,
            sunset_date=meta.sunset_date,
            version=meta.version,
            category=meta.category,
            authority_tier=get_authority_tier(meta.issuing_authority),
        )

    # ── Step 2: Add supersession edges (predecessor → successor) ─────────
    for doc_id, meta in metadata.items():
        if meta.supersedes:
            # Edge: old_doc → new_doc labelled "supersedes"
            G.add_edge(meta.supersedes, doc_id, rel="supersedes")

    # ── Step 3: Add semantic relationship edges using clause analysis ─────
    # For each pair of same-category documents, check the pre-computed
    # clause-level relationship instead of naively assuming "conflict_risk".
    doc_list = list(metadata.values())
    for i, a in enumerate(doc_list):
        for b in doc_list[i + 1:]:
            # Only process same-category pairs
            if a.category != b.category:
                continue

            # Always add same_category edge for reference
            G.add_edge(a.doc_id, b.doc_id, rel="same_category")
            G.add_edge(b.doc_id, a.doc_id, rel="same_category")

            # Check if validity windows overlap
            if not _windows_overlap(
                a.effective_date, a.sunset_date,
                b.effective_date, b.sunset_date,
            ):
                continue  # No overlap → no conflict possible

            # Look up the semantic relationship between these docs
            relationship = get_doc_pair_relationship(a.doc_id, b.doc_id)

            if relationship == "supplements":
                # Documents cover different sub-topics — NOT a conflict
                G.add_edge(a.doc_id, b.doc_id, rel="supplements")
                G.add_edge(b.doc_id, a.doc_id, rel="supplements")

            elif relationship == "narrows_scope":
                # One document is a scoped specialisation of the other
                G.add_edge(a.doc_id, b.doc_id, rel="narrows_scope")
                G.add_edge(b.doc_id, a.doc_id, rel="narrows_scope")

            elif relationship == "conflicts":
                # Genuine conflict — these docs address the same topic differently
                G.add_edge(a.doc_id, b.doc_id, rel="conflict_risk")
                G.add_edge(b.doc_id, a.doc_id, rel="conflict_risk")

            elif relationship == "supersedes":
                # Already handled by metadata supersession edges above
                pass

            else:
                # Unknown relationship — fall back to conflict_risk
                # for safety (better to flag than to miss)
                G.add_edge(a.doc_id, b.doc_id, rel="conflict_risk")
                G.add_edge(b.doc_id, a.doc_id, rel="conflict_risk")

    return G


def get_active_docs_for_date(
    metadata: dict[str, DocMeta],
    query_date: date,
) -> list[str]:
    """
    Return all doc_ids that are *in force* on query_date.
    A document is in force if:
      effective_date <= query_date  AND  (sunset_date is None OR sunset_date >= query_date)
    """
    active = []
    for doc_id, meta in metadata.items():
        if meta.effective_date <= query_date:
            if meta.sunset_date is None or meta.sunset_date >= query_date:
                active.append(doc_id)
    return active


def get_conflict_pairs(
    graph: nx.DiGraph,
    active_doc_ids: list[str],
) -> list[tuple[str, str]]:
    """
    Return unique (doc_a, doc_b) pairs from active docs that share a 'conflict_risk' edge.
    """
    seen = set()
    pairs = []
    for a in active_doc_ids:
        for b in active_doc_ids:
            if a == b:
                continue
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            if graph.has_edge(a, b) and graph[a][b].get("rel") == "conflict_risk":
                seen.add(key)
                pairs.append((a, b))
    return pairs


def get_relationship(graph: nx.DiGraph, doc_a: str, doc_b: str) -> Optional[str]:
    """
    Get the relationship type between two documents from the KG.

    Checks multiple edge types in priority order:
    supersedes > conflict_risk > narrows_scope > supplements > same_category

    Args:
        graph: The knowledge graph.
        doc_a: First document ID.
        doc_b: Second document ID.

    Returns:
        str or None: The relationship type, or None if no edge exists.
    """
    # Check edges in both directions
    for a, b in [(doc_a, doc_b), (doc_b, doc_a)]:
        if graph.has_edge(a, b):
            rel = graph[a][b].get("rel")
            if rel:
                return rel
    return None
