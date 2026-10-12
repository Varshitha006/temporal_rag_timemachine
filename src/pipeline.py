"""
pipeline.py
------------
The complete TimeMachine RAG + KG pipeline.

This is the central orchestrator that wires all modules together:

  User Question + as_of_date
        │
        ▼
  [1] Retriever           → top-k semantically similar clauses (all dates)
        │
        ▼
  [2] Temporal Filter     → filter to clauses in-force on as_of_date
                            + annotate with supersession / precedence notes
        │
        ▼
  [3] Knowledge Graph     → enrich with category relationships, find conflict-risk pairs
        │
        ▼
  [4] Conflict Detector   → three-tier precedence → decide: answer/conflict/escalate/refuse
        │
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
from src.knowledge_graph import build_knowledge_graph
from src.retriever import Retriever
from src.temporal_filter import resolve_date, temporal_filter, apply_precedence, DEFAULT_REF_DATE
from src.conflict_detector import decide, DecisionResult
from src.llm_client import LLMClient


# ── Project root corpus directory ────────────────────────────────────────────
CORPUS_DIR = Path(__file__).parent.parent / "corpus"


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


# ── Pipeline Class ────────────────────────────────────────────────────────────

class TimeMachinePipeline:
    """
    End-to-end temporal RAG + KG pipeline for policy Q&A.

    Initialisation (one-time, ~10-30 seconds):
      - Loads all 44 corpus documents
      - Builds embedding index (cached after first run)
      - Builds Knowledge Graph of document relationships
      - Initialises the Gemini LLM client
    """

    def __init__(
        self,
        corpus_dir: str | Path = CORPUS_DIR,
        top_k: int = 15,
        rebuild_index: bool = False,
    ):
        """
        Args:
            corpus_dir:    Path to the corpus/ directory.
            top_k:         Number of clauses to retrieve before temporal filtering.
            rebuild_index: If True, re-embed corpus even if cache exists.
        """
        self.top_k = top_k
        print("[Pipeline] Loading corpus...")
        self.metadata, self.clauses = load_corpus(corpus_dir)
        print(f"[Pipeline] Loaded {len(self.metadata)} documents, {len(self.clauses)} clauses.")

        print("[Pipeline] Building retriever (embedding index)...")
        self.retriever = Retriever(self.clauses, rebuild_index=rebuild_index)

        print("[Pipeline] Building Knowledge Graph...")
        self.graph = build_knowledge_graph(self.metadata)
        print(f"[Pipeline] KG built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

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
            as_of_date:  ISO date string or date object. None → 2025-06-01 (current).
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
            print(f"{'='*60}")

        # ── Step 2: Semantic Retrieval ────────────────────────────────────
        raw_results = self.retriever.search(question, k=self.top_k)
        if verbose:
            print(f"\n[Step 1 - Retrieval] Top {self.top_k} clauses retrieved:")
            for score, clause in raw_results[:5]:
                print(f"  {clause.clause_id:12s}  sim={score:.3f}  [{clause.meta.doc_id} {clause.meta.version}]")

        # ── Step 3: Temporal Filter ───────────────────────────────────────
        # keep_expired_for_context=True so expired docs appear in log (T1 trap visibility)
        filtered = temporal_filter(raw_results, self.metadata, q_date, keep_expired_for_context=True)
        in_force = [fc for fc in filtered if fc.in_force]

        if verbose:
            print(f"\n[Step 2 - Temporal Filter] {len(in_force)} in-force / {len(filtered)} retrieved:")
            for fc in filtered[:8]:
                status = "OK" if fc.in_force else "XX"
                print(f"  [{status}] {fc.clause.clause_id:12s}  {fc.precedence_note[:60]}")

        # ── Step 4: Precedence Annotation ────────────────────────────────
        annotated = apply_precedence(filtered, self.metadata)

        # ── Step 5: Conflict Detection + Decision Gate ────────────────────
        decision = decide(annotated, self.metadata, q_date, question=question)

        if verbose:
            print(f"\n[Step 3 - Decision] Label: {decision.label.upper()}")
            for log_line in decision.decision_log:
                print(f"  LOG: {log_line}")

        # ── Step 6: LLM Answer Generation ────────────────────────────────
        if skip_llm:
            # Use deterministic structured hint without calling external LLM API
            answer_text = f"[{decision.label.upper()}] {decision.answer_hint}"
        else:
            answer_text = self.llm.generate(
                question=question,
                query_date_str=str(q_date),
                decision=decision,
            )

        if verbose:
            print(f"\n[Step 4 - Answer]\n{answer_text}\n")

        # ── Collect Citations ─────────────────────────────────────────────
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
