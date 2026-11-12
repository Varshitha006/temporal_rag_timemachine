"""
tests/test_knowledge_graph.py
------------------------------
Acceptance tests for knowledge_graph.py V3.

Tests verify the five acceptance criteria from the project spec:
  1. Exactly 16 "supersedes" edges exist in the MultiDiGraph.
  2. D014 transitively supersedes D012 (but NOT the reverse).
  3. D034 and D042 share a "conflict_risk" edge.
  4. Expansion from {D014} on 2024-09-01 finds D044, but NOT on 2024-07-15.
  5. Expansion from {D013} on 2024-07-01 finds D014.

Run with:
  python -m pytest tests/test_knowledge_graph.py -v
"""

import sys
from pathlib import Path
from datetime import date

import pytest

# Add project root to sys.path so we can import src.*
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus_loader import load_corpus
from src.knowledge_graph import (
    build_knowledge_graph,
    is_doc_in_force,
    is_superseding_path,
    get_supersessor,
    expand_neighbours,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def kg():
    """
    Build the KG once for all tests in this module.
    scope="module" means we only call load_corpus + build_knowledge_graph once,
    which is much faster (avoids rebuilding the graph per test).
    """
    metadata, _clauses = load_corpus(ROOT / "corpus")
    graph = build_knowledge_graph(metadata)
    return graph, metadata


# ── Test 1: Exactly 16 "supersedes" edges exist ───────────────────────────────

def test_supersedes_edge_count(kg):
    """
    The metadata.csv lists 16 supersedes pointers (D002, D003, D006, D007,
    D010, D013, D014, D019, D022, D024, D028, D030, D031, D036, D037, D038).
    Each should produce exactly one "supersedes" edge in the MultiDiGraph.
    """
    graph, _ = kg
    # Collect all edges whose KEY is "supersedes"
    supersedes_edges = [
        (u, v) for u, v, k in graph.edges(keys=True) if k == "supersedes"
    ]
    # Print for debugging (pytest shows this on failure)
    print(f"\nSupersedes edges found: {len(supersedes_edges)}")
    for u, v in sorted(supersedes_edges):
        print(f"  {u} → {v}")

    assert len(supersedes_edges) == 16, (
        f"Expected 16 supersedes edges, got {len(supersedes_edges)}.\n"
        f"Edges: {sorted(supersedes_edges)}"
    )


# ── Test 2: D014 transitively supersedes D012 (but not the reverse) ──────────

def test_d014_transitively_supersedes_d012(kg):
    """
    Supersession chain: D012 → D013 → D014
    So D014 transitively supersedes D012 (path D012 → D013 → D014 exists).
    The reverse (D012 supersedes D014) must NOT hold.
    """
    graph, _ = kg
    # Forward: D014 supersedes D012 (transitively)
    assert is_superseding_path(graph, new_doc="D014", old_doc="D012"), (
        "D014 should transitively supersede D012 (via D013)"
    )
    # Reverse: D012 does NOT supersede D014
    assert not is_superseding_path(graph, new_doc="D012", old_doc="D014"), (
        "D012 should NOT supersede D014"
    )


# ── Test 3: D034 ↔ D042 has a "conflict_risk" edge ───────────────────────────

def test_d034_d042_conflict_risk_edge(kg):
    """
    D034 (DataClassification) and D042 (SalesEnablement) are cross-category
    conflicting documents.  Both directions should have a 'conflict_risk' edge.

    In a MultiDiGraph, graph.has_edge(u, v, key) checks whether a specific
    keyed edge exists between u and v.
    """
    graph, _ = kg
    # Both directions of the conflict_risk edge must exist
    has_fwd = graph.has_edge("D034", "D042", "conflict_risk")
    has_bwd = graph.has_edge("D042", "D034", "conflict_risk")
    assert has_fwd, "Expected D034 → D042 conflict_risk edge in KG"
    assert has_bwd, "Expected D042 → D034 conflict_risk edge in KG"


# ── Test 4a: Expansion from {D014} on 2024-09-01 finds D044 ──────────────────

def test_expansion_d014_finds_d044_on_2024_09_01(kg):
    """
    D044 (Field Ops Device Exemption) has a conflict_risk edge with D014
    (InfoSec MFA Policy).  D044's effective_date is 2024-08-01 and it has
    no sunset_date, so it IS in force on 2024-09-01.

    Expansion from {D014} should therefore include D044.
    """
    graph, metadata = kg
    query_date = date(2024, 9, 1)
    expansions = expand_neighbours(
        graph=graph,
        seed_doc_ids={"D014"},
        query_date=query_date,
        metadata=metadata,
        max_hops=2,
    )
    expanded_doc_ids = {exp.doc_id for exp in expansions}
    print(f"\nExpansion from D014 on {query_date}: {expanded_doc_ids}")
    assert "D044" in expanded_doc_ids, (
        f"Expected D044 in expansion from D014 on {query_date}, got: {expanded_doc_ids}"
    )


# ── Test 4b: Expansion from {D014} does NOT find D044 on 2024-07-15 ──────────

def test_expansion_d014_no_d044_before_effective_date(kg):
    """
    D044's effective_date is 2024-08-01.
    On 2024-07-15 (before D044 is in force), expansion should NOT include D044.
    """
    graph, metadata = kg
    query_date = date(2024, 7, 15)
    expansions = expand_neighbours(
        graph=graph,
        seed_doc_ids={"D014"},
        query_date=query_date,
        metadata=metadata,
        max_hops=2,
    )
    expanded_doc_ids = {exp.doc_id for exp in expansions}
    print(f"\nExpansion from D014 on {query_date}: {expanded_doc_ids}")
    assert "D044" not in expanded_doc_ids, (
        f"D044 should NOT be in expansion from D014 on {query_date} "
        f"(D044 not yet in force). Got: {expanded_doc_ids}"
    )


# ── Test 5: Expansion from {D013} on 2024-07-01 finds D014 ──────────────────

def test_expansion_d013_finds_d014(kg):
    """
    D013 is superseded by D014 (D013 → D014 is a "supersedes" edge).
    D014's effective_date is 2024-07-01, so on 2024-07-01 it IS in force.

    Expansion from {D013} (the old version) should follow the "supersedes"
    edge forward and return D014 (the current version) as a "governing" doc.
    """
    graph, metadata = kg
    query_date = date(2024, 7, 1)
    expansions = expand_neighbours(
        graph=graph,
        seed_doc_ids={"D013"},
        query_date=query_date,
        metadata=metadata,
        max_hops=2,
    )
    expanded_doc_ids = {exp.doc_id for exp in expansions}
    print(f"\nExpansion from D013 on {query_date}: {expanded_doc_ids}")
    assert "D014" in expanded_doc_ids, (
        f"Expected D014 in expansion from D013 on {query_date}, got: {expanded_doc_ids}"
    )
    # Also verify D014 is tagged as "governing" (supersession successor)
    governing = [exp for exp in expansions if exp.doc_id == "D014"]
    assert any(e.role == "governing" for e in governing), (
        f"D014 found in expansion but not tagged as 'governing'. Roles: {[e.role for e in governing]}"
    )


# ── Bonus: is_doc_in_force correctness ───────────────────────────────────────

def test_is_doc_in_force_d044(kg):
    """
    D044: effective 2024-08-01, no sunset.
    In force on 2024-08-01 and 2024-09-01, but NOT on 2024-07-31.
    """
    graph, _ = kg
    assert is_doc_in_force(graph, "D044", date(2024, 8, 1)),  "D044 should be in force on its effective_date"
    assert is_doc_in_force(graph, "D044", date(2024, 9, 1)),  "D044 should be in force after effective_date"
    assert not is_doc_in_force(graph, "D044", date(2024, 7, 31)), "D044 should NOT be in force before effective_date"


def test_is_doc_in_force_d013_sunset(kg):
    """
    D013: effective 2023-01-01, sunset 2024-06-30.
    In force on 2024-06-30 (inclusive) but NOT on 2024-07-01.
    """
    graph, _ = kg
    assert is_doc_in_force(graph, "D013", date(2024, 6, 30)), "D013 should be in force on its sunset date (inclusive)"
    assert not is_doc_in_force(graph, "D013", date(2024, 7, 1)),  "D013 should NOT be in force after sunset date"
