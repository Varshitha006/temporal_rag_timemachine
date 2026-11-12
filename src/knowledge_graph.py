"""
knowledge_graph.py
------------------
Builds a Knowledge Graph (KG) of policy document relationships using NetworkX.

V3: Uses MultiDiGraph so that a node-pair can carry MORE THAN ONE edge type
simultaneously.  For example, D013→D014 has both a "supersedes" edge (from
metadata) AND a "same_category" edge (same InfoSec family).  In the old
DiGraph the second add_edge call silently overwrote the first, which meant
every supersedes edge was lost.

Nodes  = documents (doc_id) — each with metadata attributes
Edges  = keyed by relationship name:
  "supersedes"     : B supersedes A  →  edge A → B  (16 edges from metadata)
  "same_category"  : A and B are in the same policy family
  "supplements"    : same category but covers a different sub-topic (no conflict)
  "narrows_scope"  : one doc is a scoped specialisation of the other
  "conflict_risk"  : docs that genuinely conflict on the same topic

Public helpers consumed by temporal_filter, conflict_detector, and pipeline:
  is_doc_in_force(graph, doc_id, query_date) → bool
  get_supersessor(graph, doc_id)             → Optional[str]   (O(1))
  is_superseding_path(graph, new, old)       → bool            (graph traversal)
  get_conflict_pairs(graph, active_doc_ids)  → list[tuple]
  get_relationship(graph, doc_a, doc_b)      → Optional[str]
  expand_neighbours(graph, seed_doc_ids,     → list[ExpandedDoc]
                    query_date, metadata, ...)
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import networkx as nx

from src.corpus_loader import DocMeta
from src.clause_relationship import (
    get_doc_pair_relationship,
    DOC_PAIR_RELATIONSHIPS,   # curated override dict
    DOC_SCOPE_KEYWORDS,       # used to auto-derive narrows_scope edges
)


# ── Authority Tier Ranking ─────────────────────────────────────────────────
# Lower number = higher authority.  Used as tie-breaker signal in conflict
# reports — NOT used to auto-resolve conflicts silently.
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
    """Return numeric tier for an issuing authority (1 = highest)."""
    for key, tier in AUTHORITY_TIERS.items():
        # Substring match in both directions handles abbreviations
        if key.lower() in authority.lower() or authority.lower() in key.lower():
            return tier
    return 6  # unknown / lowest priority


# ── Date overlap helper (used only during graph construction) ───────────────

def _windows_overlap(
    eff1: date, sun1: Optional[date],
    eff2: date, sun2: Optional[date],
) -> bool:
    """Return True if [eff1, sun1] and [eff2, sun2] overlap (inclusive)."""
    FAR_FUTURE = date(9999, 12, 31)
    end1 = sun1 if sun1 is not None else FAR_FUTURE
    end2 = sun2 if sun2 is not None else FAR_FUTURE
    return eff1 <= end2 and eff2 <= end1


# ── Shared in-force check (single source of truth) ─────────────────────────

def is_doc_in_force(
    graph: nx.MultiDiGraph,
    doc_id: str,
    query_date: date,
) -> bool:
    """
    Return True if doc_id is in force on query_date.

    Reads effective_date and sunset_date directly from the KG node so that
    temporal_filter.py and conflict_detector.py share one implementation
    (no duplicated date logic anywhere else).

    Inclusive on both ends:
      effective_date <= query_date <= sunset_date  (or no sunset)
    """
    if doc_id not in graph.nodes:
        return False
    attrs = graph.nodes[doc_id]
    # The node's effective_date must not be in the future
    if attrs["effective_date"] > query_date:
        return False
    # If a sunset_date exists, the doc must not have expired
    if attrs["sunset_date"] is not None and attrs["sunset_date"] < query_date:
        return False
    return True


# ── Supersessor lookup (O(1) via edge traversal) ───────────────────────────

def get_supersessor(graph: nx.MultiDiGraph, doc_id: str) -> Optional[str]:
    """
    Return the doc_id that DIRECTLY supersedes doc_id, or None.

    Uses the "supersedes" edge in the MultiDiGraph: if A→B has a
    "supersedes" key, then B supersedes A.

    Time complexity: O(out-degree) which is at most 1 for supersession.
    Much cheaper than the old O(N) metadata scan.
    """
    for successor in graph.successors(doc_id):
        # Check if any edge between doc_id → successor is labelled "supersedes"
        edges = graph[doc_id][successor]          # dict keyed by edge key
        if "supersedes" in edges:
            return successor
    return None


# ── Transitive supersession via graph path ─────────────────────────────────

def is_superseding_path(
    graph: nx.MultiDiGraph,
    new_doc: str,
    old_doc: str,
) -> bool:
    """
    Return True if new_doc transitively supersedes old_doc.

    Builds a view containing ONLY "supersedes" edges and then asks
    NetworkX whether a directed path old_doc → new_doc exists.
    This replaces the old while-loop metadata walk in conflict_detector.
    """
    # Create a sub-graph view with only "supersedes" typed edges
    supersedes_view = nx.subgraph_view(
        graph,
        filter_edge=lambda u, v, k: k == "supersedes",
    )
    return nx.has_path(supersedes_view, old_doc, new_doc)


# ── Build the Knowledge Graph ──────────────────────────────────────────────

def build_knowledge_graph(metadata: dict[str, DocMeta]) -> nx.MultiDiGraph:
    """
    Build and return a MultiDiGraph from the metadata dict.

    Three-stage edge construction:
      Stage A — Metadata supersession (16 directed edges, never overwritten
                 by same_category because MultiDiGraph supports multi-edges)
      Stage B — Same-category edges + curated semantic relationships
                 from DOC_PAIR_RELATIONSHIPS (overrides the "unknown" fallback)
      Stage C — Cross-category conflict_risk edges derived from the curated
                 dict (e.g. D034 ↔ D042)

    Node attributes stored per doc_id:
      title, issuing_authority, effective_date, sunset_date,
      version, category, authority_tier
    """
    G = nx.MultiDiGraph()

    # ── Stage A: Add all document nodes ─────────────────────────────────
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

    # ── Stage B: Supersession edges from metadata (the 16 "supersedes" ──
    # pointers listed in metadata.csv).  These come FIRST so they are
    # never silently overwritten (MultiDiGraph keeps all keys).
    for doc_id, meta in metadata.items():
        if meta.supersedes:
            # Direction: old_doc → new_doc  (predecessor → successor)
            G.add_edge(meta.supersedes, doc_id, key="supersedes", rel="supersedes")

    # ── Stage C: Same-category semantic relationships ────────────────────
    # For each pair of same-category documents we first check the curated
    # DOC_PAIR_RELATIONSHIPS dict (most accurate), falling back to the
    # "unknown" classification only for pairs not curated.
    #
    # Additionally: if a document is in DOC_SCOPE_KEYWORDS and the other
    # document is in the same category but NOT in DOC_SCOPE_KEYWORDS, we
    # auto-derive a "narrows_scope" edge (the scoped doc narrows the base).
    doc_list = list(metadata.values())
    for i, a in enumerate(doc_list):
        for b in doc_list[i + 1:]:

            # Always add same_category edge for same-category pairs
            if a.category == b.category:
                G.add_edge(a.doc_id, b.doc_id, key="same_category", rel="same_category")
                G.add_edge(b.doc_id, a.doc_id, key="same_category", rel="same_category")

                # Skip pairs with no temporal overlap (no conflict possible)
                if not _windows_overlap(
                    a.effective_date, a.sunset_date,
                    b.effective_date, b.sunset_date,
                ):
                    continue

                # Look up curated relationship first
                relationship = get_doc_pair_relationship(a.doc_id, b.doc_id)

                if relationship == "supplements":
                    G.add_edge(a.doc_id, b.doc_id, key="supplements", rel="supplements")
                    G.add_edge(b.doc_id, a.doc_id, key="supplements", rel="supplements")

                elif relationship == "narrows_scope":
                    # Prefer metadata: scoped doc → base doc direction
                    if a.doc_id in DOC_SCOPE_KEYWORDS:
                        G.add_edge(b.doc_id, a.doc_id, key="narrows_scope", rel="narrows_scope")
                    elif b.doc_id in DOC_SCOPE_KEYWORDS:
                        G.add_edge(a.doc_id, b.doc_id, key="narrows_scope", rel="narrows_scope")
                    else:
                        # Bidirectional when direction is ambiguous
                        G.add_edge(a.doc_id, b.doc_id, key="narrows_scope", rel="narrows_scope")
                        G.add_edge(b.doc_id, a.doc_id, key="narrows_scope", rel="narrows_scope")

                elif relationship == "conflicts":
                    # Genuine conflict — these docs address the same topic differently
                    G.add_edge(a.doc_id, b.doc_id, key="conflict_risk", rel="conflict_risk")
                    G.add_edge(b.doc_id, a.doc_id, key="conflict_risk", rel="conflict_risk")

                elif relationship == "supersedes":
                    # Already handled by metadata supersession edges above; skip
                    pass

                else:
                    # "unknown" pair: auto-detect narrows_scope from DOC_SCOPE_KEYWORDS
                    # if exactly one doc has scope keywords, otherwise use conflict_risk
                    a_scoped = a.doc_id in DOC_SCOPE_KEYWORDS
                    b_scoped = b.doc_id in DOC_SCOPE_KEYWORDS
                    if a_scoped and not b_scoped:
                        G.add_edge(b.doc_id, a.doc_id, key="narrows_scope", rel="narrows_scope")
                    elif b_scoped and not a_scoped:
                        G.add_edge(a.doc_id, b.doc_id, key="narrows_scope", rel="narrows_scope")
                    else:
                        # Two unknown same-category overlapping docs → flag conservatively
                        G.add_edge(a.doc_id, b.doc_id, key="conflict_risk", rel="conflict_risk")
                        G.add_edge(b.doc_id, a.doc_id, key="conflict_risk", rel="conflict_risk")

    # ── Stage D: Cross-category conflict_risk edges ──────────────────────
    # The curated dict contains cross-category pairs (different category
    # fields in metadata).  These would be skipped by the same-category
    # loop above but are real conflicts (e.g. D034 DataClassification ↔
    # D042 SalesEnablement).
    for (doc_a, doc_b), rel in DOC_PAIR_RELATIONSHIPS.items():
        if rel != "conflicts":
            continue
        if doc_a not in metadata or doc_b not in metadata:
            continue
        # Only add if they are in DIFFERENT categories (same-cat already handled)
        if metadata[doc_a].category != metadata[doc_b].category:
            G.add_edge(doc_a, doc_b, key="conflict_risk", rel="conflict_risk")
            G.add_edge(doc_b, doc_a, key="conflict_risk", rel="conflict_risk")

    return G


# ── Active document query ──────────────────────────────────────────────────

def get_active_docs_for_date(
    graph: nx.MultiDiGraph,
    query_date: date,
) -> list[str]:
    """Return all doc_ids that are in force on query_date (from graph nodes)."""
    return [
        doc_id for doc_id in graph.nodes
        if is_doc_in_force(graph, doc_id, query_date)
    ]


# ── Conflict pair listing ──────────────────────────────────────────────────

def get_conflict_pairs(
    graph: nx.MultiDiGraph,
    active_doc_ids: list[str],
) -> list[tuple[str, str]]:
    """
    Return unique (doc_a, doc_b) pairs from active docs that share a
    'conflict_risk' edge in the MultiDiGraph.
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    active_set = set(active_doc_ids)

    for a in active_doc_ids:
        for b in graph.successors(a):
            if b not in active_set or b == a:
                continue
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            # Check if any edge between a→b is "conflict_risk"
            edges = graph[a][b]
            if "conflict_risk" in edges:
                seen.add(key)
                pairs.append((a, b))
    return pairs


# ── Single relationship lookup ─────────────────────────────────────────────

def get_relationship(
    graph: nx.MultiDiGraph,
    doc_a: str,
    doc_b: str,
) -> Optional[str]:
    """
    Get the HIGHEST-PRIORITY relationship between two documents.

    Priority: supersedes > conflict_risk > narrows_scope > supplements > same_category

    Checks both directions of the edge.
    """
    PRIORITY = ["supersedes", "conflict_risk", "narrows_scope", "supplements", "same_category"]

    for a, b in [(doc_a, doc_b), (doc_b, doc_a)]:
        if not graph.has_node(a) or not graph.has_node(b):
            continue
        if b not in graph[a]:
            continue
        edges = graph[a][b]  # dict of {key: attr_dict}
        for rel in PRIORITY:
            if rel in edges:
                return rel
    return None


# ── Adjacent-document expansion ────────────────────────────────────────────

@dataclass
class ExpandedDoc:
    """
    A document pulled in by KG edge traversal (not by similarity search).
    Carries the edge type and the source doc that caused the expansion.
    """
    doc_id: str          # the newly added document
    via_doc: str         # which seed doc caused this expansion
    edge_type: str       # "supersedes", "conflict_risk", or "narrows_scope"
    role: str            # "governing" (supersession successor) or "context" (conflict/scope)


def expand_neighbours(
    graph: nx.MultiDiGraph,
    seed_doc_ids: set[str],
    query_date: date,
    metadata: dict[str, DocMeta],
    max_hops: int = 2,
    scope_question: Optional[str] = None,
) -> list[ExpandedDoc]:
    """
    Follow KG edges from seed_doc_ids to find related docs that similarity
    search may have missed.

    Expansion rules:
      1. "supersedes" forward: old_doc → new_doc.  If a retrieved doc has
         been superseded, pull in the successor (the governing version).
      2. "conflict_risk" neighbours: docs that conflict with a retrieved doc
         (context-only; cited only if they actually appear in conflict report).
      3. "narrows_scope" neighbours: docs that narrow/override a retrieved doc.

    Filters:
      - Skip docs already in seed_doc_ids.
      - Skip docs NOT in force on query_date.
      - For scoped docs (in DOC_SCOPE_KEYWORDS), skip if the question
        doesn't target their scope.

    Args:
        graph:          The MultiDiGraph KG.
        seed_doc_ids:   Doc IDs already found by semantic retrieval.
        query_date:     The as-of date for temporal validity.
        metadata:       Full metadata dict (for scope keyword checks).
        max_hops:       BFS depth limit (default 2).
        scope_question: The user's question text, used for scope filtering.

    Returns:
        List of ExpandedDoc objects (deduplicated, in BFS order).
    """
    from src.clause_relationship import DOC_SCOPE_KEYWORDS, query_targets_scoped_doc

    # Edge types we follow during expansion
    EXPANSION_EDGES = {"supersedes", "conflict_risk", "narrows_scope"}

    # BFS state
    frontier: set[str] = set(seed_doc_ids)  # docs seen at any hop
    results: list[ExpandedDoc] = []
    current_frontier = set(seed_doc_ids)    # docs to explore this hop

    for _hop in range(max_hops):
        next_frontier: set[str] = set()

        for seed in current_frontier:
            if seed not in graph.nodes:
                continue

            # Explore all successors of this seed
            for neighbour in graph.successors(seed):
                if neighbour in frontier:
                    continue  # already seen

                # Determine which edge type connects seed → neighbour
                edges = graph[seed][neighbour]
                edge_type = None
                for rel in ["supersedes", "conflict_risk", "narrows_scope"]:
                    if rel in edges:
                        edge_type = rel
                        break
                if edge_type is None:
                    continue  # only same_category / supplements edges → skip

                # Must be in force on the query date
                if not is_doc_in_force(graph, neighbour, query_date):
                    continue

                # Scope filter: if the neighbour is a scoped doc and the
                # question doesn't target its scope, skip it
                if scope_question and neighbour in DOC_SCOPE_KEYWORDS:
                    if not query_targets_scoped_doc(scope_question, neighbour):
                        continue

                # "governing" role only for supersession successors
                role = "governing" if edge_type == "supersedes" else "context"

                results.append(ExpandedDoc(
                    doc_id=neighbour,
                    via_doc=seed,
                    edge_type=edge_type,
                    role=role,
                ))
                frontier.add(neighbour)
                next_frontier.add(neighbour)

        if not next_frontier:
            break   # no new neighbours found — stop early
        current_frontier = next_frontier

    return results


# ── Utility: list active docs on a date ────────────────────────────────────

def get_active_docs_for_date(
    graph: nx.MultiDiGraph,
    query_date: date,
) -> list[str]:
    """
    Return a sorted list of doc_ids that are in force on query_date.
    Used by visualize_kg.py to display an active-doc count.
    """
    return sorted(
        doc_id for doc_id in graph.nodes
        if is_doc_in_force(graph, doc_id, query_date)
    )

