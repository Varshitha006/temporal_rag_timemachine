"""
corpus_loader.py
----------------
Loads all corpus documents (D001.md - D044.md) and the metadata.csv.

Each document has:
  - YAML front-matter: doc_id, title, issuing_authority, effective_date,
    sunset_date, supersedes, version, category
  - Numbered clauses: D0xx:C1, D0xx:C2, ...

This module returns a list of Clause objects (the atomic retrieval unit)
and a dict of DocMeta objects (for temporal filtering and KG construction).
"""

import csv
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml


# ── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class DocMeta:
    """Metadata for one policy document (mirrors YAML front-matter)."""
    doc_id: str
    title: str
    issuing_authority: str
    effective_date: date
    sunset_date: Optional[date]      # None means still current
    supersedes: Optional[str]        # doc_id of the predecessor, or None
    version: str
    category: str


@dataclass
class Clause:
    """A single numbered clause — the atomic retrieval unit."""
    clause_id: str          # e.g. "D003:C1"
    doc_id: str             # e.g. "D003"
    text: str               # raw clause text
    meta: DocMeta           # back-reference to the parent document metadata


# ── Helpers ─────────────────────────────────────────────────────────────────

def _parse_date(raw) -> Optional[date]:
    """Convert an ISO string or None/NaN to a date (or None)."""
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "null", "none", "nan")):
        return None
    if isinstance(raw, date):
        return raw
    return date.fromisoformat(str(raw).strip())


def _extract_clauses(doc_text: str, meta: DocMeta) -> list[Clause]:
    """
    Parse numbered clauses from the markdown body.
    Pattern: **D0xx:C<n>** — <text>
    """
    # Match clause IDs like **D003:C1** — followed by text until next clause or EOF
    pattern = re.compile(
        r"\*\*(" + re.escape(meta.doc_id) + r":C\d+)\*\*\s*[—–-]\s*(.*?)(?=\*\*" + re.escape(meta.doc_id) + r":C\d+\*\*|\Z)",
        re.DOTALL,
    )
    clauses = []
    for m in pattern.finditer(doc_text):
        clause_id = m.group(1).strip()
        text = m.group(2).strip()
        clauses.append(Clause(clause_id=clause_id, doc_id=meta.doc_id, text=text, meta=meta))
    return clauses


# ── Public API ───────────────────────────────────────────────────────────────

def load_corpus(corpus_dir: str | Path) -> tuple[dict[str, DocMeta], list[Clause]]:
    """
    Load all .md files in corpus_dir and return:
      - metadata: dict mapping doc_id → DocMeta
      - clauses:  list of all Clause objects across all documents
    """
    corpus_dir = Path(corpus_dir)
    metadata_path = corpus_dir / "metadata.csv"

    # 1. Load metadata from CSV for fast, authoritative field access
    metadata: dict[str, DocMeta] = {}
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["doc_id"].strip()
            metadata[doc_id] = DocMeta(
                doc_id=doc_id,
                title=row["title"].strip(),
                issuing_authority=row["issuing_authority"].strip(),
                effective_date=_parse_date(row["effective_date"]),
                sunset_date=_parse_date(row.get("sunset_date")),
                supersedes=row["supersedes"].strip() or None,
                version=row["version"].strip(),
                category=row["category"].strip(),
            )

    # 2. Parse clause text from each .md file
    all_clauses: list[Clause] = []
    for md_file in sorted(corpus_dir.glob("D*.md")):
        raw = md_file.read_text(encoding="utf-8")

        # Extract YAML front-matter between --- delimiters
        fm_match = re.match(r"^---\s*\n(.*?)\n---", raw, re.DOTALL)
        doc_id_from_file = None
        if fm_match:
            fm = yaml.safe_load(fm_match.group(1))
            doc_id_from_file = fm.get("doc_id", "").strip()

        doc_id = doc_id_from_file or md_file.stem
        if doc_id not in metadata:
            # Build meta on the fly from front-matter if somehow missing from CSV
            if fm_match:
                fm = yaml.safe_load(fm_match.group(1))
                metadata[doc_id] = DocMeta(
                    doc_id=doc_id,
                    title=fm.get("title", doc_id),
                    issuing_authority=fm.get("issuing_authority", "Unknown"),
                    effective_date=_parse_date(fm.get("effective_date")),
                    sunset_date=_parse_date(fm.get("sunset_date")),
                    supersedes=fm.get("supersedes") or None,
                    version=fm.get("version", "v1.0"),
                    category=fm.get("category", "Unknown"),
                )

        meta = metadata[doc_id]
        clauses = _extract_clauses(raw, meta)
        all_clauses.extend(clauses)

    return metadata, all_clauses
