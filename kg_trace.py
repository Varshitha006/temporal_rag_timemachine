"""
kg_trace.py
-----------
Acceptance Criterion 5: "KG usage trace"

For one specific query, this script prints EVERY place the KG was consulted:
  - Temporal check (is_doc_in_force on graph node attributes)
  - Supersessor lookup via edge traversal
  - KG expansion (edges followed to find adjacent docs)
  - Supersession path check (Tier-1 in conflict detector)
  - Conflict edge traversal (conflict_risk edges)

Also demonstrates the "search missed it" ablation (Acceptance Criterion 3):
  run the pipeline with D044 removed from retriever output (simulating RAG miss),
  and show that expansion=ON recovers the correct answer while expansion=OFF fails.

Usage:
  python kg_trace.py
"""

import sys
from pathlib import Path
from datetime import date

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.corpus_loader import load_corpus
from src.knowledge_graph import (
    build_knowledge_graph,
    is_doc_in_force,
    get_supersessor,
    is_superseding_path,
    expand_neighbours,
    get_conflict_pairs,
)
from src.temporal_filter import resolve_date, temporal_filter, apply_precedence
from src.conflict_detector import decide

# -- Load corpus and build KG once --------------------------------------------─
print("Loading corpus & KG...")
metadata, clauses = load_corpus(ROOT / "corpus")
kg = build_knowledge_graph(metadata)

QUERY = "Are field engineers using shared kiosk devices required to use MFA?"
AS_OF = date(2024, 9, 1)

print(f"\n{'='*70}")
print(f"KG USAGE TRACE")
print(f"Question: {QUERY}")
print(f"As-of:    {AS_OF}")
print(f"{'='*70}\n")

# -- Trace 1: Temporal check (is_doc_in_force on graph node attributes) --------
print("-- TRACE 1: Temporal checks (graph node attributes) --")
for doc_id in ["D012", "D013", "D014", "D044"]:
    in_force = is_doc_in_force(kg, doc_id, AS_OF)
    attrs = kg.nodes[doc_id]
    print(
        f"  is_doc_in_force(KG, {doc_id}, {AS_OF}) = {in_force} "
        f"  [{attrs['effective_date']} – {attrs.get('sunset_date') or 'open'}]"
    )

# -- Trace 2: Supersessor lookup via edge traversal ----------------------------─
print("\n-- TRACE 2: Supersessor lookup via 'supersedes' edges --")
for doc_id in ["D012", "D013", "D014"]:
    succ = get_supersessor(kg, doc_id)
    print(f"  get_supersessor(KG, {doc_id}) = {succ}")

# -- Trace 3: KG expansion from {D014} on AS_OF ------------------------------─
print("\n-- TRACE 3: KG expansion from {D014} --")
expansions = expand_neighbours(
    graph=kg,
    seed_doc_ids={"D014"},
    query_date=AS_OF,
    metadata=metadata,
    max_hops=2,
    scope_question=QUERY,
)
for exp in expansions:
    print(f"  {exp.via_doc} --{exp.edge_type}--> {exp.doc_id}  role={exp.role}")

# -- Trace 4: Supersession path check (Tier-1 conflict detector) --------------─
print("\n-- TRACE 4: Supersession path checks (Tier-1 in _analyse_pair) --")
for new, old in [("D014", "D012"), ("D014", "D013"), ("D012", "D014")]:
    result = is_superseding_path(kg, new_doc=new, old_doc=old)
    print(f"  is_superseding_path(KG, new={new}, old={old}) = {result}")

# -- Trace 5: Conflict edge traversal ----------------------------------------─
print("\n-- TRACE 5: Conflict edge traversal for active docs --")
active = [doc_id for doc_id in kg.nodes if is_doc_in_force(kg, doc_id, AS_OF)]
print(f"  Active docs on {AS_OF}: {sorted(active)}")
conflict_pairs = get_conflict_pairs(kg, active)
print(f"  Conflict pairs from KG: {conflict_pairs}")
for a, b in conflict_pairs:
    print(f"    KG: {a} --conflict_risk-- {b}")

# -- Trace 6: Full decide() trace with graph ----------------------------------─
print("\n-- TRACE 6: Full pipeline decide() with KG (for query) --")
from src.retriever import Retriever
retriever = Retriever(clauses, rebuild_index=False)
raw = retriever.search(QUERY, k=15)
filtered = temporal_filter(raw, metadata, AS_OF, keep_expired_for_context=True, graph=kg)
annotated = apply_precedence(filtered, metadata)
decision = decide(annotated, metadata, AS_OF, question=QUERY, graph=kg)

print(f"\n  Decision label: {decision.label.upper()}")
print(f"  Decision log:")
for line in decision.decision_log:
    prefix = "  KG>" if "KG:" in line else "    >"
    print(f"  {prefix} {line}")

# -- ABLATION: "search missed it" demonstration --------------------------------
print(f"\n{'='*70}")
print("ABLATION: 'Search missed it' — D044 removed from retriever output")
print(f"{'='*70}")

# Simulate search that misses D044
raw_no_d044 = [(s, c) for s, c in raw if c.doc_id != "D044"]

# WITHOUT expansion
filtered_no_exp = temporal_filter(raw_no_d044, metadata, AS_OF, keep_expired_for_context=True, graph=kg)
annotated_no_exp = apply_precedence(filtered_no_exp, metadata)
dec_no_exp = decide(annotated_no_exp, metadata, AS_OF, question=QUERY, graph=None)  # no graph = no expansion
print(f"\n  expansion=OFF, D044 missing → label: {dec_no_exp.label}")
print(f"  Governing docs: {sorted({fc.clause.doc_id for fc in dec_no_exp.governing_clauses})}")

# WITH expansion (graph passed, D014 seed will find D044 via conflict_risk edge)
filtered_exp = temporal_filter(raw_no_d044, metadata, AS_OF, keep_expired_for_context=False, graph=kg)
in_force_seeds = {fc.clause.doc_id for fc in filtered_exp if fc.in_force}
expansions_ablation = expand_neighbours(
    graph=kg,
    seed_doc_ids=in_force_seeds,
    query_date=AS_OF,
    metadata=metadata,
    max_hops=2,
    scope_question=QUERY,
)
# Add expansion clauses back
expansion_clauses = [(0.0, c) for c in clauses if c.doc_id in {e.doc_id for e in expansions_ablation}]
filtered_combined = filtered_exp + temporal_filter(expansion_clauses, metadata, AS_OF, keep_expired_for_context=False, graph=kg)
annotated_exp = apply_precedence(filtered_combined, metadata)
dec_exp = decide(annotated_exp, metadata, AS_OF, question=QUERY, graph=kg)

print(f"\n  expansion=ON,  D044 recovered via KG → label: {dec_exp.label}")
expanded_doc_ids = {e.doc_id for e in expansions_ablation}
print(f"  KG expanded: {sorted(expanded_doc_ids)}")
print(f"  Governing docs: {sorted({fc.clause.doc_id for fc in dec_exp.governing_clauses})}")

print(f"\n{'='*70}")
print("VERDICT:")
if "D044" in {fc.clause.doc_id for fc in dec_exp.governing_clauses} or dec_exp.label == "conflict":
    print("  [OK] expansion=ON correctly surfaces D044 (or detects conflict with it).")
else:
    print("  [!!]  expansion=ON did not surface D044 — check scope filtering.")
if dec_no_exp.label != dec_exp.label:
    print(f"  [OK] Label changed: '{dec_no_exp.label}' (no expansion) → '{dec_exp.label}' (with expansion).")
else:
    print(f"  [i]  Same label both ways: '{dec_no_exp.label}' — may not be a label-level difference for this query.")
print(f"{'='*70}")
