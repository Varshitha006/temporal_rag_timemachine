"""
pipeline.py
------------
V3: KG-integrated TimeMachine RAG pipeline.

This is the central orchestrator that wires all modules together:

  User Question + as_of_date
        │
        ▼
  [1] Retriever           → top-k semantically similar clauses (all dates)
        │
        ▼
  [2] Temporal Filter     → filter to clauses in-force on as_of_date
                            + annotate with supersession / precedence notes
        │   (graph passed for O(1) supersessor lookup)
        ▼
  [3] KG Expansion        → follow edges from retrieved docs to find related
                            docs that semantic search missed (use_graph_expansion)
        │
        ▼
  [4] Conflict Detector   → three-tier precedence → decide: answer/conflict/escalate/refuse
        │   (graph passed so relationships are read from edges, not hard-coded dict)
        ▼
  [5] LLM (Gemini)        → generate final natural-language answer with citations
        │
        ▼
  Answer + Decision Log + Citations

Usage:
    from src.pipeline import TimeMachinePipeline
    pipeline = TimeMachinePipeline()
    result   = pipeline.query("What is the WFH policy?", as_of_date="2024-03-01")
    print(result["answer"])
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from src.corpus_loader import Clause, DocMeta, load_corpus
from src.knowledge_graph import (
    build_knowledge_graph,
    expand_neighbours,
    ExpandedDoc,
    is_doc_in_force,
)
from src.retriever import Retriever
from src.temporal_filter import (
    resolve_date,
    temporal_filter,
    apply_precedence,
    DEFAULT_REF_DATE,
    is_in_force,
)
from src.conflict_detector import decide, DecisionResult
from src.llm_client import LLMClient


# ── Project root corpus directory ────────────────────────────────────────────
CORPUS_DIR = Path(__file__).parent.parent / "corpus"

# ── How many top retrieved docs to use as seeds for KG expansion ─────────────
EXPANSION_SEED_N = 5


# ── Pipeline Output ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Full output of one pipeline query."""
    question: str                          # original user question
    as_of_date: str                        # ISO date used (or "null → 2025-06-01")
    label: str                             # answer | conflict | escalate | refuse
    answer: str                            # final LLM-generated answer
    citations: list[str]                   # list of clause_ids cited (e.g. ["D003:C1"])
    decision_log: list[str]                # step-by-step reasoning trace
    retrieved_count: int                   # how many clauses were retrieved
    in_force_count: int                    # how many survived temporal filter
    expanded_docs: list[str]               # doc_ids added by KG expansion (empty if disabled)


# ── Pipeline Class ────────────────────────────────────────────────────────────

class TimeMachinePipeline:
    """
    End-to-end temporal RAG + KG pipeline for policy Q&A.

    Initialisation (one-time, ~10-30 seconds):
      - Loads all 44 corpus documents
      - Builds embedding index (cached after first run)
      - Builds Knowledge Graph of document relationships (MultiDiGraph)
      - Initialises the Gemini LLM client

    Key flag:
      use_graph_expansion (default True): when False, KG expansion is skipped
      entirely and the pipeline behaves identically to V2 (useful for ablation).
    """

    def __init__(
        self,
        corpus_dir: str | Path = CORPUS_DIR,
        top_k: int = 15,
        rebuild_index: bool = False,
        use_graph_expansion: bool = True,
    ):
        """
        Args:
            corpus_dir:          Path to the corpus/ directory.
            top_k:               Number of clauses to retrieve before temporal filtering.
            rebuild_index:       If True, re-embed corpus even if cache exists.
            use_graph_expansion: If True, follow KG edges to pull in adjacent docs
                                  that semantic retrieval missed.
        """
        self.top_k = top_k
        self.use_graph_expansion = use_graph_expansion

        print("[Pipeline] Loading corpus...")
        self.metadata, self.clauses = load_corpus(corpus_dir)
        print(f"[Pipeline] Loaded {len(self.metadata)} documents, {len(self.clauses)} clauses.")

        print("[Pipeline] Building retriever (embedding index)...")
        self.retriever = Retriever(self.clauses, rebuild_index=rebuild_index)

        print("[Pipeline] Building Knowledge Graph...")
        self.graph = build_knowledge_graph(self.metadata)

        # Summarise what the KG actually contains — helps verify correctness
        n_supersedes = sum(
            1 for u, v, k in self.graph.edges(keys=True) if k == "supersedes"
        )
        n_conflict = sum(
            1 for u, v, k in self.graph.edges(keys=True) if k == "conflict_risk"
        )
        print(
            f"[Pipeline] KG built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges total "
            f"({n_supersedes} supersedes, {n_conflict} conflict_risk)."
        )

        print("[Pipeline] Initialising LLM client...")
        self.llm = LLMClient()

        print("[Pipeline] Ready!\n")

    # ── Main Query Method ─────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        as_of_date: Optional[str | date] = None,
        verbose: bool = False,
        skip_llm: bool = False,
    ) -> PipelineResult:
        """
        Run the full pipeline for a user question.

        Args:
            question:    Natural language question about a policy.
            as_of_date:  ISO date string or date object. None → 2025-06-01.
            verbose:     Print step-by-step trace to stdout.
            skip_llm:    If True, skip calling LLM and use structured hint (faster for eval).

        Returns:
            PipelineResult with answer, citations, label, and decision log.
        """
        # ── Step 1: Resolve query date ─────────────────────────────────────
        q_date = resolve_date(as_of_date)
        date_str = f"{as_of_date} → {q_date}" if as_of_date is None else str(q_date)

        if verbose:
            print(f"\n{'='*60}")
            print(f"QUESTION : {question}")
            print(f"AS_OF    : {date_str}")
            print(f"EXPANSION: {'ON' if self.use_graph_expansion else 'OFF'}")
            print(f"{'='*60}")

        # ── Step 2: Semantic Retrieval ────────────────────────────────────
        raw_results = self.retriever.search(question, k=self.top_k)
        if verbose:
            print(f"\n[Step 1 - Retrieval] Top {self.top_k} clauses retrieved:")
            for score, clause in raw_results[:5]:
                print(f"  {clause.clause_id:12s}  sim={score:.3f}  [{clause.meta.doc_id} {clause.meta.version}]")

        # ── Step 3: Temporal Filter (graph-accelerated) ────────────────────
        # Pass graph so _find_supersessor uses O(1) edge lookup
        filtered = temporal_filter(
            raw_results, self.metadata, q_date,
            keep_expired_for_context=True,
            graph=self.graph,
        )
        in_force = [fc for fc in filtered if fc.in_force]

        if verbose:
            print(f"\n[Step 2 - Temporal Filter] {len(in_force)} in-force / {len(filtered)} retrieved:")
            for fc in filtered[:8]:
                status = "OK" if fc.in_force else "XX"
                print(f"  [{status}] {fc.clause.clause_id:12s}  {fc.precedence_note[:60]}")

        # ── Step 4: KG Expansion (optional) ───────────────────────────────
        expanded_docs: list[str] = []

        if self.use_graph_expansion and in_force:
            # Take the unique doc_ids from top-EXPANSION_SEED_N in-force clauses
            seen_for_seed: set[str] = set()
            seed_doc_ids: set[str] = set()
            for fc in in_force:
                if fc.clause.doc_id not in seen_for_seed:
                    seen_for_seed.add(fc.clause.doc_id)
                    seed_doc_ids.add(fc.clause.doc_id)
                    if len(seed_doc_ids) >= EXPANSION_SEED_N:
                        break

            if verbose:
                print(f"\n[Step 3 - KG Expansion] Seeds: {sorted(seed_doc_ids)}")

            # Follow KG edges to find adjacent docs
            expansions: list[ExpandedDoc] = expand_neighbours(
                graph=self.graph,
                seed_doc_ids=seed_doc_ids,
                query_date=q_date,
                metadata=self.metadata,
                max_hops=2,
                scope_question=question,
            )

            if verbose and expansions:
                print(f"[Step 3 - KG Expansion] Found {len(expansions)} adjacent doc(s):")
                for exp in expansions:
                    print(f"  {exp.via_doc} --{exp.edge_type}--> {exp.doc_id} [{exp.role}]")

            if expansions:
                # Build a set of clause_ids already in the filtered list
                already_in = {fc.clause.clause_id for fc in filtered}

                # Fetch ALL clauses for expanded docs that were not yet in the result set
                new_clauses_by_doc: dict[str, list[Clause]] = {}
                for exp in expansions:
                    if exp.doc_id not in new_clauses_by_doc:
                        doc_clauses = [
                            c for c in self.clauses
                            if c.doc_id == exp.doc_id
                            and c.clause_id not in already_in
                        ]
                        if doc_clauses:
                            new_clauses_by_doc[exp.doc_id] = doc_clauses
                            expanded_docs.append(exp.doc_id)

                # Run temporal_filter on the new clauses (score = 0.0 as placeholder)
                expansion_raw = [(0.0, c) for doc_clauses in new_clauses_by_doc.values() for c in doc_clauses]
                if expansion_raw:
                    expansion_filtered = temporal_filter(
                        expansion_raw, self.metadata, q_date,
                        keep_expired_for_context=False,
                        graph=self.graph,
                    )
                    # Only keep in-force ones from expansion
                    expansion_in_force = [fc for fc in expansion_filtered if fc.in_force]

                    # Tag each expanded clause's precedence_note with how it arrived
                    exp_by_doc = {exp.doc_id: exp for exp in expansions}
                    for fc in expansion_in_force:
                        exp = exp_by_doc.get(fc.clause.doc_id)
                        if exp:
                            fc.precedence_note += (
                                f" | KG-expansion via {exp.via_doc} "
                                f"--{exp.edge_type}-- {exp.doc_id} [{exp.role}]"
                            )

                    # Append expanded clauses to the filtered set
                    filtered = filtered + expansion_in_force

                    if verbose:
                        print(f"[Step 3 - KG Expansion] Added {len(expansion_in_force)} clause(s) from expansion.")

        # ── Step 5: Precedence Annotation ─────────────────────────────────
        annotated = apply_precedence(filtered, self.metadata)

        # ── Step 6: Conflict Detection + Decision Gate (KG-integrated) ────
        # Pass graph so decide() reads conflict edges from the graph
        decision = decide(annotated, self.metadata, q_date, question=question, graph=self.graph)

        if verbose:
            print(f"\n[Step 4 - Decision] Label: {decision.label.upper()}")
            for log_line in decision.decision_log:
                print(f"  LOG: {log_line}")

        # ── Step 7: LLM Answer Generation ─────────────────────────────────
        if skip_llm:
            answer_text = f"[{decision.label.upper()}] {decision.answer_hint}"
        else:
            answer_text = self.llm.generate(
                question=question,
                query_date_str=str(q_date),
                decision=decision,
            )

        if verbose:
            print(f"\n[Step 5 - Answer]\n{answer_text}\n")

        # ── Collect Citations ──────────────────────────────────────────────
        citations = list({fc.clause.clause_id for fc in decision.governing_clauses})
        citations.sort()

        return PipelineResult(
            question=question,
            as_of_date=date_str,
            label=decision.label,
            answer=answer_text,
            citations=citations,
            decision_log=decision.decision_log,
            retrieved_count=len(raw_results),
            in_force_count=len(in_force),
            expanded_docs=expanded_docs,
        )

    # ── Batch Query ───────────────────────────────────────────────────────────

    def batch_query(
        self,
        questions: list[dict],
        verbose: bool = False,
    ) -> list[PipelineResult]:
        """
        Run the pipeline for a list of question dicts.
        Each dict must have 'question' key, optionally 'as_of_date'.

        Args:
            questions: List of dicts with keys: question, as_of_date (optional).
            verbose:   Print per-question trace.

        Returns:
            List of PipelineResult objects.
        """
        results = []
        for i, q in enumerate(questions, 1):
            print(f"[Batch] Processing {i}/{len(questions)}: {q['question'][:60]}...")
            result = self.query(
                question=q["question"],
                as_of_date=q.get("as_of_date"),
                verbose=verbose,
            )
            results.append(result)
        return results
