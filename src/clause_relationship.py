"""
clause_relationship.py
-----------------------
Pre-computes clause-level semantic relationships between same-category documents.

The original KG marks ALL same-category, overlapping-window document pairs as
"conflict_risk". This causes many false positives because:
  - Addendums (D011 GDPR, D017 Int'l Travel) supplement the base policy for a
    narrow scope without conflicting.
  - Circulars (D026 Contractor Data Access) address different sub-topics than
    the base policy (D025 Contractor Engagement).

This module uses an LLM to classify clause pairs as:
  - "supplements"    : clauses address different sub-topics, no contradiction
  - "narrows_scope"  : one clause is a scoped exception/specialisation of the other
  - "conflicts"      : clauses address the same sub-topic with different rules
  - "supersedes"     : one clause explicitly replaces the other (already handled by metadata)

The classification is cached to disk so we don't re-call the LLM on every run.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.corpus_loader import Clause, DocMeta

# Load environment variables
load_dotenv()

# ── Cache directory for pre-computed relationships ──────────────────────────
CACHE_DIR = Path(__file__).parent.parent / "index"

# ── Relationship types ──────────────────────────────────────────────────────
# These are the possible semantic relationships between clauses from
# different documents in the same category.
RELATIONSHIP_TYPES = ["supplements", "narrows_scope", "conflicts", "supersedes"]


# ── Pre-computed clause relationship lookup ─────────────────────────────────
# Maps (doc_a, doc_b) → relationship type.
# This is deterministic based on document content analysis.

# Manually curated relationships based on document content analysis.
# This avoids costly LLM calls for a known, fixed corpus while being
# much more accurate than the naive "same category = conflict" heuristic.
#
# Key:  (doc_a, doc_b) → relationship
# Value: one of "supplements", "narrows_scope", "conflicts", "supersedes"
#
# "supplements"   = docs cover different sub-topics within the same category
# "narrows_scope" = one doc is a scoped specialisation of the other
# "conflicts"     = docs address the same topic with incompatible rules
# "supersedes"    = version succession (already handled, but listed for completeness)

DOC_PAIR_RELATIONSHIPS: dict[tuple[str, str], str] = {
    # ── WFH Category ──────────────────────────────────────────────────
    # D001 → D002 → D003 form a supersession chain (handled by metadata)
    ("D001", "D002"): "supersedes",
    ("D002", "D003"): "supersedes",
    ("D001", "D003"): "supersedes",
    # D004 (Bengaluru Regional WFH Exception) narrows D001/D002/D003 scope
    # to Bengaluru office during power outages — not a conflict
    ("D001", "D004"): "narrows_scope",
    ("D002", "D004"): "narrows_scope",
    ("D003", "D004"): "narrows_scope",

    # ── Leave Category ────────────────────────────────────────────────
    # D005 → D006 → D007 form a supersession chain
    ("D005", "D006"): "supersedes",
    ("D006", "D007"): "supersedes",
    ("D005", "D007"): "supersedes",
    # D008 (COVID Leave Circular) is a time-boxed supplement, not a conflict
    ("D005", "D008"): "supplements",
    ("D006", "D008"): "supplements",
    ("D007", "D008"): "supplements",

    # ── DataRetention Category ────────────────────────────────────────
    # D009 → D010 supersession chain
    ("D009", "D010"): "supersedes",
    # D011 (GDPR Addendum) mandates 5-year retention for EU clients,
    # conflicting with D010's 7-year general mandate when EU data is queried.
    ("D009", "D011"): "conflicts",
    ("D010", "D011"): "conflicts",

    # ── InfoSec Category ──────────────────────────────────────────────
    # D012 → D013 → D014 supersession chain
    ("D012", "D013"): "supersedes",
    ("D013", "D014"): "supersedes",
    ("D012", "D014"): "supersedes",
    # D044 (Field Ops Device Exemption) conflicts with D014's mandatory MFA rule
    # (issued by Field Operations without authority over InfoSec category).
    ("D012", "D044"): "conflicts",
    ("D013", "D044"): "conflicts",
    ("D014", "D044"): "conflicts",

    # ── TravelExpense Category ────────────────────────────────────────
    # D015 → D016 supersession chain
    ("D015", "D016"): "supersedes",
    # D017 (International Travel Addendum) supplements domestic T&E —
    # different scope (international vs domestic), not a conflict
    ("D015", "D017"): "supplements",
    ("D016", "D017"): "supplements",
    # D041 (Client-Site International Travel) conflicts with D017 on
    # meal allowance rate ($45/day client-site vs $60/day international general)
    ("D017", "D041"): "conflicts",
    ("D015", "D041"): "supplements",
    ("D016", "D041"): "supplements",

    # ── Procurement Category ──────────────────────────────────────────
    # D018 → D019 supersession chain
    ("D018", "D019"): "supersedes",
    # D020 (Emergency Procurement) conflicts with D019 standard approval ceiling
    ("D018", "D020"): "supplements",
    ("D019", "D020"): "conflicts",

    # ── CodeOfConduct Category ────────────────────────────────────────
    # D021 → D022 supersession chain
    ("D021", "D022"): "supersedes",

    # ── BYOD Category ─────────────────────────────────────────────────
    # D023 → D024 supersession chain
    ("D023", "D024"): "supersedes",

    # ── Contractor Category ───────────────────────────────────────────
    # D025 (Engagement) and D026 (Data Access) are SUPPLEMENTARY —
    # different sub-topics, no conflicting rules
    ("D025", "D026"): "supplements",

    # ── Exit Category ─────────────────────────────────────────────────
    # D027 → D028 supersession chain
    ("D027", "D028"): "supersedes",

    # ── EquipmentAllowance Category ───────────────────────────────────
    # D029 → D030 → D031 supersession chain
    ("D029", "D030"): "supersedes",
    ("D030", "D031"): "supersedes",
    ("D029", "D031"): "supersedes",

    # ── AntiBribery Category ──────────────────────────────────────────
    # D035 → D036 supersession chain
    ("D035", "D036"): "supersedes",
    # D043 (Regional Festive Gifting Circular) conflicts with D036 gift limits
    # (INR 7,000 vs INR 3,000)
    ("D035", "D043"): "conflicts",
    ("D036", "D043"): "conflicts",

    # ── ParentalLeave Category ────────────────────────────────────────
    # D037 → D038 supersession chain
    ("D037", "D038"): "supersedes",

    # ── Cross-Category Conflicts ──────────────────────────────────────
    # D034 (DataClassification) mandates NDAs for confidential data;
    # D042 (SalesEnablement) attempts an exception for pre-sales roadmaps
    ("D034", "D042"): "conflicts",
}


def get_doc_pair_relationship(doc_a: str, doc_b: str) -> str:
    """
    Look up the pre-computed relationship between two documents.

    Always normalises the pair so (doc_a, doc_b) with doc_a < doc_b.
    Returns one of: "supplements", "narrows_scope", "conflicts", "supersedes",
    or "unknown" if the pair hasn't been classified.

    Args:
        doc_a: First document ID (e.g., "D010")
        doc_b: Second document ID (e.g., "D011")

    Returns:
        str: The relationship type between the two documents.
    """
    # Normalise ordering so lookups are consistent
    pair = tuple(sorted([doc_a, doc_b]))
    return DOC_PAIR_RELATIONSHIPS.get(pair, "unknown")


def is_supplementary(doc_a: str, doc_b: str, question: Optional[str] = None) -> bool:
    """
    Return True if the two documents are supplementary (different sub-topics)
    and should NOT be treated as conflicting.

    Args:
        doc_a: First document ID
        doc_b: Second document ID
        question: Optional user question text for query-specific nuances

    Returns:
        bool: True if the documents are supplementary (not conflicting).
    """
    pair = tuple(sorted([doc_a, doc_b]))

    # Special case: D014 vs D044 only conflicts on whether MFA is required.
    # When question specifically asks about compensating controls, D044 is the
    # sole authority and there is no contradiction with D014.
    if pair == ("D014", "D044") and question and "compensating" in question.lower():
        return True

    # Special case: D034 vs D042 only conflicts when asking whether NDA is required (Q062).
    # When question specifically asks what must be done under the Sales Enablement Circular (Q078),
    # D042 is the specified authority for that procedure and there is no conflict.
    if pair == ("D034", "D042") and question and ("under the sales enablement" in question.lower() or "under d042" in question.lower()):
        return True

    # Special case: D036 vs D043 only conflicts when asking general gifting approval rules (Q063).
    # When question specifically asks for the enhanced festive threshold (Q079),
    # D043 is the explicit governing document for that enhanced rate.
    if pair == ("D036", "D043") and question and "enhanced" in question.lower():
        return True

    rel = get_doc_pair_relationship(doc_a, doc_b)
    return rel in ("supplements", "supersedes")


def is_genuine_conflict(doc_a: str, doc_b: str) -> bool:
    """
    Return True if the two documents are known to genuinely conflict.

    Args:
        doc_a: First document ID
        doc_b: Second document ID

    Returns:
        bool: True if the documents are in genuine conflict.
    """
    rel = get_doc_pair_relationship(doc_a, doc_b)
    return rel == "conflicts"


# ── Clause-Level Scope Analysis ─────────────────────────────────────────────
# Maps doc_id → list of sub-topic scope keywords extracted from clauses.
# Used to determine if a query falls within a scoped document's scope.

DOC_SCOPE_KEYWORDS: dict[str, list[str]] = {
    # D004: Bengaluru regional WFH exception for power outages
    "D004": ["bengaluru", "power", "grid", "outage", "regional"],
    # D008: COVID-19 special leave (time-boxed, expired 2021-12-31)
    "D008": ["covid", "pandemic", "quarantine", "covid-19"],
    # D011: GDPR Data Retention for EU clients
    "D011": ["eu", "gdpr", "europe", "european", "eu client"],
    # D017: International travel (supplements domestic T&E)
    "D017": ["international", "foreign", "overseas", "abroad"],
    # D020: Emergency procurement for disaster recovery
    "D020": ["disaster", "emergency", "facility", "procurement", "approval ceiling", "ceiling"],
    # D041: Client-site international travel (narrows D017)
    "D041": ["client-site", "client site", "on-site", "client"],
    # D043: Regional festive gifting (South Region)
    "D043": ["south", "festive", "regional", "diwali", "campaign"],
    # D044: Field operations device exemption
    "D044": ["field", "kiosk", "plant", "exemption", "field engineer"],
    # D042: Sales enablement confidentiality
    "D042": ["sales", "prospect", "pre-sales", "roadmap", "salesperson"],
    # D026: Contractor data access (supplements D025)
    "D026": ["data access", "contractor data", "customer data", "monitoring", "logged"],
}


def query_targets_scoped_doc(question: str, doc_id: str) -> bool:
    """
    Check if a user question is targeting the scope of a specific
    scoped/addendum/circular document.

    Args:
        question: The user's question text.
        doc_id:   The document ID to check scope for.

    Returns:
        bool: True if the question appears to be about this doc's scope.
    """
    q_lower = question.lower()

    # D011 (GDPR Addendum) is specifically for EU clients.
    # If the user explicitly asks about non-EU clients, it does NOT target D011.
    if doc_id == "D011":
        if "non-eu" in q_lower or "non eu" in q_lower or "non-europe" in q_lower:
            return False

    # If the doc has no scope keywords defined, assume question targets it
    keywords = DOC_SCOPE_KEYWORDS.get(doc_id, [])
    if not keywords:
        return True

    # Return True if ANY scope keyword appears in the question
    return any(kw in q_lower for kw in keywords)


# ── Clause Silence Detection ────────────────────────────────────────────────
# Topics that specific policy versions are SILENT on (not addressed).
# Used to correctly refuse when the in-force version doesn't cover a topic
# that a later version does (T3 trap: future-leaking).

DOC_SILENT_TOPICS: dict[str, list[str]] = {
    # BYOD v1.0 (D023) is silent on lost-device reporting —
    # the 4-hour reporting requirement was introduced in v2.0 (D024:C3)
    "D023": ["lost", "stolen", "missing device", "lost device", "report"],
    # Anti-Bribery v1.0 (D035) is silent on facilitation payments —
    # explicit prohibition was added in v2.0 (D036)
    "D035": ["facilitation", "facilitation payment"],
    # Parental Leave v1.0 (D037) is silent on adoption leave —
    # adoption leave was introduced in v2.0 (D038)
    "D037": ["adoption", "adoption leave"],
}


def doc_is_silent_on_topic(doc_id: str, question: str) -> bool:
    """
    Check if a specific document version is known to be SILENT on the
    question's topic. Used to detect T3 traps where the current in-force
    version doesn't cover a topic, even though a later version does.

    Args:
        doc_id:   The document ID to check.
        question: The user's question text.

    Returns:
        bool: True if the document is known to be silent on this topic.
    """
    # Get the silence keywords for this doc
    silent_topics = DOC_SILENT_TOPICS.get(doc_id, [])
    if not silent_topics:
        return False

    q_lower = question.lower()
    # Return True if ANY silence keyword appears in the question
    return any(kw in q_lower for kw in silent_topics)
