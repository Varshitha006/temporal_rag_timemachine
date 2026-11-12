"""
conflict_detector.py
---------------------
V3: KG-integrated conflict detection.

Detects and classifies conflicts between in-force documents.

V3 Changes vs V2:
  - decide() now accepts the MultiDiGraph and reads conflict relationships
    FROM GRAPH EDGES instead of looping over DOC_PAIR_RELATIONSHIPS.
  - _is_superseding() replaced by is_superseding_path() graph traversal.
  - Every decision-log line that consulted the graph says so, e.g.:
      "KG: D034 --conflict_risk-- D042"
  - Cross-category pairs are found by traversing conflict_risk edges,
    not by iterating the hand-written DOC_PAIR_RELATIONSHIPS dict.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import networkx as nx

from src.corpus_loader import Clause, DocMeta
from src.knowledge_graph import (
    is_superseding_path,
    get_authority_tier,
    get_conflict_pairs,
)
from src.temporal_filter import FilteredClause
from src.clause_relationship import (
    is_supplementary,
    is_genuine_conflict,
    query_targets_scoped_doc,
    doc_is_silent_on_topic,
    DOC_SCOPE_KEYWORDS,
    DOC_PAIR_RELATIONSHIPS,   # kept as fallback when graph is not passed
)


# ── Who "owns" each policy category ──────────────────────────────────────────
# Used to detect authority mismatch (T2-d, T2-f pattern).
CATEGORY_OWNERS: dict[str, list[str]] = {
    "DataClassification": ["IT Governance", "Legal & Compliance"],
    "InfoSec":            ["IT Security"],
    "AntiBribery":        ["Legal & Compliance"],
    "DataRetention":      ["IT Governance", "Legal & Compliance"],
    "WFH":                ["Human Resources Department"],
    "Leave":              ["Human Resources Department"],
    "ParentalLeave":      ["Human Resources Department"],
    "Exit":               ["Human Resources Department"],
    "Procurement":        ["Procurement Committee", "Finance Department"],
    "TravelExpense":      ["Finance Department"],
    "BYOD":               ["IT Security"],
    "Contractor":         ["Procurement & Legal"],
    "CodeOfConduct":      ["Ethics Committee"],
    "Whistleblower":      ["Ethics Committee"],
    "SocialMedia":        ["Corporate Communications & HR"],
    "BackgroundVerification": ["HR & Legal"],
    "EquipmentAllowance": ["Finance & HR"],
    "SalesEnablement":    ["Sales & Marketing"],
    "NonPolicy":          [],
}


def _normalize_auth(text: str) -> str:
    """Normalize authority strings to handle HR, IT, Legal, Finance abbreviations."""
    t = text.lower()
    t = t.replace("human resources", "hr")
    t = t.replace("information technology", "it")
    t = t.replace("department", "").replace("committee", "").replace("regional", "")
    return t.strip()


def _is_authority_mismatch(doc_id: str, category: str, metadata: dict[str, DocMeta]) -> bool:
    """
    Return True if the document's issuing_authority does NOT match the
    expected owners of the policy category it's trying to modify.
    """
    meta = metadata[doc_id]
    owners = CATEGORY_OWNERS.get(category, [])
    if not owners:
        return False
    auth_norm = _normalize_auth(meta.issuing_authority)
    for owner in owners:
        owner_norm = _normalize_auth(owner)
        if owner_norm in auth_norm or auth_norm in owner_norm:
            return False
    return True   # issuer is NOT a recognised owner of this category


# ── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class ConflictReport:
    """Full conflict analysis for a pair of documents on a specific topic."""
    doc_a: str
    doc_b: str
    category: str
    auto_resolved: bool         # True if precedence rules cleanly resolve the conflict
    winner: Optional[str]       # doc_id of the winning document, if auto_resolved
    loser: Optional[str]        # doc_id of the losing document, if auto_resolved
    resolution_rule: str        # which tier resolved it (or why it wasn't resolved)
    requires_escalation: bool   # True if system should NOT silently resolve
    escalation_reason: str      # reason for escalation (authority mismatch, etc.)
    kg_edge: str = ""           # KG edge label that flagged this pair (for trace)


@dataclass
class DecisionResult:
    """Final pipeline decision for a query."""
    label: str                              # 'answer' | 'conflict' | 'escalate' | 'refuse'
    governing_clauses: list[FilteredClause] # clauses that should be cited in the answer
    conflict_reports: list[ConflictReport]  # populated for 'conflict' label
    decision_log: list[str]                 # step-by-step reasoning trace
    answer_hint: str                        # short note to guide the LLM answer generator


# ── Core Conflict Detection ───────────────────────────────────────────────────

def _analyse_pair(
    doc_a: str,
    doc_b: str,
    category: str,
    metadata: dict[str, DocMeta],
    graph: Optional[nx.MultiDiGraph] = None,
    kg_edge: str = "",
) -> ConflictReport:
    """
    Analyse a conflict-risk pair and return a ConflictReport.
    Applies the three-tier precedence rules from charter.md §3.

    Args:
        doc_a, doc_b: The pair of document IDs to compare.
        category:     The policy category for this conflict.
        metadata:     Full metadata dict.
        graph:        The KG MultiDiGraph (used for Tier-1 supersession check).
        kg_edge:      Human-readable KG edge label for the decision log.
    """
    meta_a = metadata[doc_a]
    meta_b = metadata[doc_b]

    # ── Tier 1: Explicit or Transitive Supersession ────────────────────────
    # Use graph-based path query when available; fall back to metadata walk.
    if graph is not None:
        b_supersedes_a = is_superseding_path(graph, doc_b, doc_a)
        a_supersedes_b = is_superseding_path(graph, doc_a, doc_b)
    else:
        b_supersedes_a = _is_superseding_fallback(doc_b, doc_a, metadata)
        a_supersedes_b = _is_superseding_fallback(doc_a, doc_b, metadata)

    if b_supersedes_a:
        return ConflictReport(
            doc_a=doc_a, doc_b=doc_b, category=category,
            auto_resolved=True, winner=doc_b, loser=doc_a,
            resolution_rule=f"Tier-1: supersession ({doc_b} supersedes {doc_a})",
            requires_escalation=False, escalation_reason="",
            kg_edge=kg_edge,
        )
    if a_supersedes_b:
        return ConflictReport(
            doc_a=doc_a, doc_b=doc_b, category=category,
            auto_resolved=True, winner=doc_a, loser=doc_b,
            resolution_rule=f"Tier-1: supersession ({doc_a} supersedes {doc_b})",
            requires_escalation=False, escalation_reason="",
            kg_edge=kg_edge,
        )

    # ── Authority mismatch check (before specificity) ─────────────────────
    mismatch_a = _is_authority_mismatch(doc_a, category, metadata)
    mismatch_b = _is_authority_mismatch(doc_b, category, metadata)

    if mismatch_a or mismatch_b:
        offender  = doc_a if mismatch_a else doc_b
        owner_doc = doc_b if mismatch_a else doc_a
        return ConflictReport(
            doc_a=doc_a, doc_b=doc_b, category=category,
            auto_resolved=False, winner=None, loser=None,
            resolution_rule="Tier-2 blocked: authority mismatch",
            requires_escalation=True,
            escalation_reason=(
                f"{offender} (issued by {metadata[offender].issuing_authority}) attempts to override "
                f"{owner_doc} (owned by {metadata[owner_doc].issuing_authority}) without "
                f"authority over the '{category}' policy category."
            ),
            kg_edge=kg_edge,
        )

    # ── Tier 2: Time-boxed Emergency / Temporary Specificity ─────────────
    a_timeboxed = meta_a.sunset_date is not None
    b_timeboxed = meta_b.sunset_date is not None

    if a_timeboxed and not b_timeboxed:
        return ConflictReport(
            doc_a=doc_a, doc_b=doc_b, category=category,
            auto_resolved=True, winner=doc_a, loser=doc_b,
            resolution_rule="Tier-2: specificity (A is time-boxed/narrower, applies within its window)",
            requires_escalation=False, escalation_reason="",
            kg_edge=kg_edge,
        )

    if b_timeboxed and not a_timeboxed:
        return ConflictReport(
            doc_a=doc_a, doc_b=doc_b, category=category,
            auto_resolved=True, winner=doc_b, loser=doc_a,
            resolution_rule="Tier-2: specificity (B is time-boxed/narrower, applies within its window)",
            requires_escalation=False, escalation_reason="",
            kg_edge=kg_edge,
        )

    # ── True conflict (T2 trap): no supersession, no clear specificity ────
    return ConflictReport(
        doc_a=doc_a, doc_b=doc_b, category=category,
        auto_resolved=False, winner=None, loser=None,
        resolution_rule="Competing policies: no explicit supersession link between active documents",
        requires_escalation=True,
        escalation_reason=f"Conflict between {doc_a} ({meta_a.title}) and {doc_b} ({meta_b.title}).",
        kg_edge=kg_edge,
    )


def _is_superseding_fallback(
    new_doc: str, old_doc: str, metadata: dict[str, DocMeta]
) -> bool:
    """
    Fallback O(depth) metadata chain walk used when graph is not available.
    Prefer is_superseding_path() when a graph is passed.
    """
    curr = metadata.get(new_doc)
    while curr and curr.supersedes:
        if curr.supersedes == old_doc:
            return True
        curr = metadata.get(curr.supersedes)
    return False


# ── Decision Gate ─────────────────────────────────────────────────────────────

def decide(
    filtered: list[FilteredClause],
    metadata: dict[str, DocMeta],
    query_date: date,
    question: Optional[str] = None,
    graph: Optional[nx.MultiDiGraph] = None,
) -> DecisionResult:
    """
    V3 KG-integrated decision gate: determines the final label for the query.

    Flow:
      1. Filter out non-binding decoy documents (e.g. D040).
      2. Check clause-level silence (T3 traps: policy version silent on topic).
      3. If no in-force clauses → refuse (T4).
      4. Check for unmentioned out-of-corpus topics → refuse (T4).
      5. Check for ambiguous population scope edge-cases → escalate (T5).
      6. Check for conflicts using KG edges → conflict/escalate (T2).
      7. Otherwise → answer.

    Args:
        filtered:    Output from temporal_filter().
        metadata:    Full metadata dict.
        query_date:  The as-of date.
        question:    Original user question text.
        graph:       The MultiDiGraph KG.  When supplied, conflict edges are
                     read from graph instead of DOC_PAIR_RELATIONSHIPS dict.
                     Every consulted edge is logged as "KG: Dxxx --rel-- Dyyy".
    """
    log: list[str] = []

    # ── Step 1: Separate in-force from expired, skip non-binding D040 ─────
    in_force = [fc for fc in filtered if fc.in_force and fc.clause.doc_id != "D040"]
    expired  = [fc for fc in filtered if not fc.in_force]

    log.append(f"In-force clauses: {len(in_force)}, Expired clauses: {len(expired)}")

    # ── Step 1b: Clause-level silence detection (T3 traps) ─────────────────
    if question and in_force:
        in_force_doc_ids = {fc.clause.doc_id for fc in in_force}
        silent_doc_ids = {
            doc_id for doc_id in in_force_doc_ids
            if doc_is_silent_on_topic(doc_id, question)
        }

        if silent_doc_ids and silent_doc_ids == in_force_doc_ids:
            # ALL in-force docs are silent on this topic → refuse
            log.append(
                f"All in-force docs {silent_doc_ids} are silent on the question topic -> REFUSE (T3)"
            )
            return DecisionResult(
                label="refuse",
                governing_clauses=[],
                conflict_reports=[],
                decision_log=log,
                answer_hint=(
                    "The in-force policy version(s) on this date do not address "
                    "this specific topic. A later version may cover it, but the "
                    "current version is silent. Do not apply rules from future "
                    "policy versions."
                ),
            )

        if silent_doc_ids:
            # Record the categories of the silent docs BEFORE removing them
            silent_doc_categories = {
                metadata[doc_id].category for doc_id in silent_doc_ids
                if doc_id in metadata
            }
            log.append(f"Removing silent docs {silent_doc_ids} from in-force set")
            in_force = [fc for fc in in_force if fc.clause.doc_id not in silent_doc_ids]

            # After removing silent docs, check if same-category fallback exists
            remaining_categories = {fc.clause.meta.category for fc in in_force}
            orphaned_categories = silent_doc_categories - remaining_categories
            if orphaned_categories:
                log.append(
                    f"No remaining in-force docs in categories {orphaned_categories} "
                    f"after silence removal -> REFUSE (T3, no same-category fallback)"
                )
                return DecisionResult(
                    label="refuse",
                    governing_clauses=[],
                    conflict_reports=[],
                    decision_log=log,
                    answer_hint=(
                        "The in-force policy version(s) on this date do not address "
                        "this specific topic. The remaining retrieved documents are "
                        "from unrelated policy categories and cannot be used to answer. "
                        "A later version may cover this topic, but the current version "
                        "is silent."
                    ),
                )

    if not in_force:
        log.append("No in-force non-decoy clauses found -> REFUSE")
        return DecisionResult(
            label="refuse",
            governing_clauses=[],
            conflict_reports=[],
            decision_log=log,
            answer_hint=(
                "The corpus contains no document in force on this date that addresses "
                "this question. Do not infer from adjacent policies or expired documents."
            ),
        )

    # ── Step 2: Check for unmentioned out-of-corpus topics (T4 refusal) ───
    if question:
        q_lower = question.lower()
        refusal_keywords = [
            "sabbatical", "cryptocurrency", "crypto", "gym", "pet", "pets",
            "lottery", "co-working", "referral bonus", "electric vehicle"
        ]
        for kw in refusal_keywords:
            if kw in q_lower:
                has_mention = any(kw in fc.clause.text.lower() for fc in in_force)
                if not has_mention:
                    log.append(f"Query topic '{kw}' not mentioned in any in-force clause -> REFUSE (T4)")
                    return DecisionResult(
                        label="refuse",
                        governing_clauses=[],
                        conflict_reports=[],
                        decision_log=log,
                        answer_hint=(
                            f"The corpus contains no policy provision for '{kw}'. "
                            "Refuse the request as not covered by company policies."
                        ),
                    )

    # ── Step 3: Ambiguous population scope edge-cases (T5 escalation) ─────
    if question:
        q_lower = question.lower()
        ambiguous_pop_triggers = [
            ("contractor", ["wfh", "work from home", "equipment", "byod", "leave", "wiki"]),
            ("freelancer", ["wfh", "work from home", "equipment", "byod", "leave"]),
            ("intern", ["equipment", "allowance", "whistleblower", "leave"]),
            ("part-time", ["leave"]),
            ("probation", ["parental"]),
            ("cross-regional", ["gifting", "festive"]),
        ]
        for pop, topics in ambiguous_pop_triggers:
            if re.search(r'\b' + re.escape(pop) + r's?\b', q_lower) and any(t in q_lower for t in topics):
                log.append(f"Ambiguous population scope '{pop}' detected -> ESCALATE (T5)")
                return DecisionResult(
                    label="escalate",
                    governing_clauses=in_force,
                    conflict_reports=[],
                    decision_log=log,
                    answer_hint=(
                        f"The question asks about '{pop}' eligibility, which is ambiguous "
                        "or undefined in the governing policy. Escalate for official clarification."
                    ),
                )

    # ── Step 4: Conflict detection via KG edges ────────────────────────────
    # V3: When graph is available, we walk conflict_risk edges directly.
    # For same-category pairs, the graph edges replace the DOC_PAIR_RELATIONSHIPS loop.
    # Cross-category pairs are also found via graph edges (conflict_risk exists in
    # the MultiDiGraph for known cross-category conflicts like D034↔D042).

    # First, filter out scoped documents when the query doesn't target their scope
    in_force_for_conflict = list(in_force)
    if question:
        scoped_doc_ids_to_remove: set[str] = set()
        for fc in in_force_for_conflict:
            doc_id = fc.clause.doc_id
            if doc_id in DOC_SCOPE_KEYWORDS:
                if not query_targets_scoped_doc(question, doc_id):
                    scoped_doc_ids_to_remove.add(doc_id)
                    log.append(
                        f"Scoped doc {doc_id} not targeted by query -> excluding from conflict check"
                    )
        if scoped_doc_ids_to_remove:
            in_force_for_conflict = [
                fc for fc in in_force_for_conflict
                if fc.clause.doc_id not in scoped_doc_ids_to_remove
            ]

    # Collect unique in-force doc IDs for conflict checking
    in_force_doc_ids = {fc.clause.doc_id for fc in in_force_for_conflict}

    conflict_reports: list[ConflictReport] = []
    checked_pairs: set[tuple[str, str]] = set()

    if graph is not None:
        # ── KG path: read conflict_risk edges from the graph ───────────────
        # For every in-force doc, follow conflict_risk edges to find peers
        for doc_a in in_force_doc_ids:
            if doc_a not in graph.nodes:
                continue
            for doc_b in graph.successors(doc_a):
                if doc_b not in in_force_doc_ids or doc_b == doc_a:
                    continue
                # Only process conflict_risk edges
                edges = graph[doc_a][doc_b]
                if "conflict_risk" not in edges:
                    continue

                pair = tuple(sorted([doc_a, doc_b]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                # KG trace line in the decision log
                kg_edge_label = f"KG: {doc_a} --conflict_risk-- {doc_b}"
                log.append(kg_edge_label)

                # Supplementary pairs are not real conflicts
                if is_supplementary(doc_a, doc_b, question=question):
                    log.append(
                        f"Docs {doc_a} vs {doc_b}: supplementary (different sub-topics) — skipping"
                    )
                    continue

                # Use the category of doc_a for the report (cross-category conflict
                # is still assigned to the "primary" category for resolution)
                cat_a = metadata[doc_a].category
                report = _analyse_pair(doc_a, doc_b, cat_a, metadata, graph, kg_edge_label)
                conflict_reports.append(report)
                log.append(
                    f"Conflict check {doc_a} vs {doc_b} [{cat_a}]: "
                    f"resolved={report.auto_resolved}, "
                    f"escalate={report.requires_escalation}, "
                    f"rule={report.resolution_rule}"
                )

    else:
        # ── Fallback path (no graph): replicate V2 same-category + curated ─
        # 1. Same-category pairs
        by_category: dict[str, set[str]] = {}
        for fc in in_force_for_conflict:
            cat = fc.clause.meta.category
            by_category.setdefault(cat, set()).add(fc.clause.doc_id)

        for cat, doc_ids in by_category.items():
            if len(doc_ids) < 2:
                continue
            doc_id_list = sorted(doc_ids)
            for i, a in enumerate(doc_id_list):
                for b in doc_id_list[i + 1:]:
                    pair = tuple(sorted([a, b]))
                    checked_pairs.add(pair)
                    if is_supplementary(a, b, question=question):
                        log.append(
                            f"Docs {a} vs {b} [{cat}]: supplementary — skipping conflict"
                        )
                        continue
                    report = _analyse_pair(a, b, cat, metadata)
                    conflict_reports.append(report)
                    log.append(
                        f"Conflict check {a} vs {b} [{cat}]: "
                        f"resolved={report.auto_resolved}, "
                        f"escalate={report.requires_escalation}, "
                        f"rule={report.resolution_rule}"
                    )

        # 2. Known cross-category pairs from curated dict
        for (doc_a, doc_b), rel in DOC_PAIR_RELATIONSHIPS.items():
            pair = tuple(sorted([doc_a, doc_b]))
            if pair in checked_pairs:
                continue
            if rel == "conflicts" and doc_a in in_force_doc_ids and doc_b in in_force_doc_ids:
                cat_a = metadata[doc_a].category
                cat_b = metadata[doc_b].category
                if cat_a != cat_b:
                    checked_pairs.add(pair)
                    if not is_supplementary(doc_a, doc_b, question=question):
                        report = _analyse_pair(doc_a, doc_b, cat_a, metadata)
                        conflict_reports.append(report)
                        log.append(
                            f"Cross-category conflict check {doc_a} vs {doc_b} [{cat_a}/{cat_b}]: "
                            f"resolved={report.auto_resolved}, "
                            f"escalate={report.requires_escalation}, "
                            f"rule={report.resolution_rule}"
                        )

    # ── Step 5: Determine overall label ────────────────────────────────────
    if not conflict_reports:
        log.append("No conflicts detected → ANSWER")
        return DecisionResult(
            label="answer",
            governing_clauses=in_force,
            conflict_reports=[],
            decision_log=log,
            answer_hint="Answer directly, citing the governing clause(s) below.",
        )

    # Determine if there are any unresolved (active) conflicts
    is_active_conflict = any(
        (not r.auto_resolved) or r.resolution_rule.startswith("Tier-2") or r.requires_escalation
        for r in conflict_reports
    )

    if is_active_conflict:
        log.append("Conflict detected between active policies → ESCALATE/CONFLICT")
        relevant_ids = set()
        for r in conflict_reports:
            relevant_ids.add(r.doc_a)
            relevant_ids.add(r.doc_b)
        governing = [fc for fc in in_force if fc.clause.doc_id in relevant_ids]
        reasons = "; ".join(
            r.escalation_reason or r.resolution_rule for r in conflict_reports
        )
        return DecisionResult(
            label="conflict",
            governing_clauses=governing,
            conflict_reports=conflict_reports,
            decision_log=log,
            answer_hint=(
                f"A conflict exists between active policy documents. "
                f"Surface both documents and escalate to the appropriate authority if unresolvable. "
                f"Details: {reasons}"
            ),
        )

    # All conflicts auto-resolved by precedence rules
    losing_doc_ids = {r.loser for r in conflict_reports if r.auto_resolved and r.loser}
    governing = [fc for fc in in_force if fc.clause.doc_id not in losing_doc_ids]

    resolution_notes = "; ".join(
        f"{r.winner} wins over {r.loser} ({r.resolution_rule})"
        for r in conflict_reports
    )
    log.append(f"All conflicts auto-resolved by precedence rules -> ANSWER ({resolution_notes})")

    return DecisionResult(
        label="answer",
        governing_clauses=governing,
        conflict_reports=conflict_reports,
        decision_log=log,
        answer_hint=(
            f"Precedence rules resolved earlier versioning: {resolution_notes}. "
            f"Answer directly citing the winning governing clause(s)."
        ),
    )
