"""
eval.py
--------
Evaluation harness for the TimeMachine pipeline.

Scores the pipeline against gold_qa_dev.jsonl on four metrics:
  1. Retrieval Hit@k     : Was the governing clause in the top-k retrieved?
  2. Label Accuracy      : Does the pipeline's label match the gold label?
  3. Refusal Accuracy    : Of 'refuse' questions, did the pipeline refuse? (not hallucinate)
  4. Conflict Detection  : Of 'conflict' questions, was a conflict detected?

Also reports per-trap-type breakdown (T1–T5) and a summary table.

Usage:
  python eval.py                          # evaluates dev set
  python eval.py --split held_out        # evaluates held-out set (end of project only)
  python eval.py --file gold_qa_dev.jsonl # use a specific file
  python eval.py --verbose               # print each question's output
"""

import os
import sys
import subprocess
import argparse
import json
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Auto-detect and re-execute with project .venv python if available and not yet active
venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and os.environ.get("TIMEMACHINE_VENV_ACTIVE") != "1":
    os.environ["TIMEMACHINE_VENV_ACTIVE"] = "1"
    try:
        sys.exit(subprocess.call([str(venv_python)] + sys.argv))
    except Exception:
        pass   # Fallback to current python if subprocess fails

from src.pipeline import TimeMachinePipeline, PipelineResult


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def compute_retrieval_hit(governing: list[str], citations: list[str]) -> bool:
    """
    Return True if AT LEAST ONE governing clause from the gold set
    appears in the pipeline's citations.
    This is Hit@k for k = pipeline.top_k.
    """
    if not governing:
        return True   # For refuse/escalate questions, no governing clause needed
    governing_set = set(governing)
    citations_set = set(citations)
    return bool(governing_set & citations_set)


def compute_label_match(gold_label: str, pred_label: str) -> bool:
    """
    Label accuracy: does the predicted label match the gold label?
    'conflict' and 'escalate' are treated as the same for label matching
    since both surface ambiguity to the user (both are non-answers).
    """
    # Normalise labels: 'escalate' maps to 'conflict', 'answer' maps to 'answerable'
    NORM = {
        "escalate": "conflict",
        "answer": "answerable",
    }
    return NORM.get(gold_label, gold_label) == NORM.get(pred_label, pred_label)


def compute_refusal_accuracy(gold_label: str, pred_label: str) -> Optional[bool]:
    """
    Only relevant for gold_label='refuse'.
    Returns True if the pipeline correctly refused (pred_label in {'refuse','escalate'}).
    Returns None for non-refuse questions.
    """
    if gold_label != "refuse":
        return None
    return pred_label in {"refuse", "escalate"}


def compute_conflict_accuracy(gold_label: str, pred_label: str) -> Optional[bool]:
    """
    Only relevant for gold_label='conflict'.
    Returns True if pipeline detected a conflict (pred_label='conflict').
    Returns None for non-conflict questions.
    """
    if gold_label != "conflict":
        return None
    return pred_label == "conflict"


# ── Import Optional (used above) ──────────────────────────────────────────────
from typing import Optional


# ── Main Evaluation Loop ──────────────────────────────────────────────────────

def run_evaluation(
    gold_file: Path,
    pipeline: TimeMachinePipeline,
    verbose: bool = False,
    skip_llm: bool = True,
) -> dict:
    """
    Run full evaluation against a gold JSONL file.

    Returns a dict of aggregate and per-trap metrics.
    """
    # Load gold questions
    gold_items = []
    with open(gold_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold_items.append(json.loads(line))

    print(f"\n[Eval] Evaluating {len(gold_items)} questions from {gold_file.name}...\n", flush=True)

    # Counters
    total = len(gold_items)
    retrieval_hits  = 0
    label_matches   = 0
    refusal_correct = 0
    refusal_total   = 0
    conflict_correct= 0
    conflict_total  = 0

    # Per-trap breakdown
    trap_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "label_match": 0, "hit": 0})

    per_question_results = []

    for item in gold_items:
        qid       = item["id"]
        question  = item["question"]
        as_of     = item.get("as_of_date")  # may be null
        gold_label= item["label"]
        governing = item.get("governing", [])
        trap_type = item.get("trap", "none")

        # Run pipeline
        result: PipelineResult = pipeline.query(
            question=question,
            as_of_date=as_of,
            verbose=verbose,
            skip_llm=skip_llm,
        )

        # Compute individual metrics
        hit   = compute_retrieval_hit(governing, result.citations)
        lmatch= compute_label_match(gold_label, result.label)
        ref   = compute_refusal_accuracy(gold_label, result.label)
        con   = compute_conflict_accuracy(gold_label, result.label)

        # Accumulate
        retrieval_hits += int(hit)
        label_matches  += int(lmatch)
        if ref is not None:
            refusal_total   += 1
            refusal_correct += int(ref)
        if con is not None:
            conflict_total   += 1
            conflict_correct += int(con)

        # Per-trap
        trap_stats[trap_type]["total"]      += 1
        trap_stats[trap_type]["label_match"]+= int(lmatch)
        trap_stats[trap_type]["hit"]        += int(hit)

        # Store result for reporting
        per_question_results.append({
            "id": qid,
            "question": question[:60],
            "gold_label": gold_label,
            "pred_label": result.label,
            "trap": trap_type,
            "hit": hit,
            "label_match": lmatch,
            "citations": result.citations,
            "answer_snippet": result.answer[:120] if result.answer else "",
        })

        # Print per-question summary using ASCII status to prevent Windows terminal encoding errors
        status = "OK" if lmatch else "ERR"
        hit_str = "HIT" if hit else "MISS"
        print(
            f"  [{status:3s}] {qid} | gold={gold_label:9s} pred={result.label:9s} "
            f"trap={trap_type:4s} | {hit_str}"
        )

    # ── Aggregate scores ──────────────────────────────────────────────────────
    scores = {
        "total_questions":       total,
        "retrieval_hit_at_k":    retrieval_hits / total,
        "label_accuracy":        label_matches  / total,
        "refusal_accuracy":      refusal_correct / refusal_total if refusal_total > 0 else None,
        "conflict_detection":    conflict_correct/ conflict_total if conflict_total > 0 else None,
        "per_trap": {
            trap: {
                "total":       stats["total"],
                "label_acc":   stats["label_match"] / stats["total"],
                "retrieval_hit": stats["hit"] / stats["total"],
            }
            for trap, stats in sorted(trap_stats.items())
        },
        "per_question": per_question_results,
    }

    return scores


def print_report(scores: dict, split_name: str):
    """Pretty-print the evaluation report."""
    print(f"\n{'='*65}")
    print(f"  TimeMachine Evaluation Report — {split_name.upper()} SET")
    print(f"{'='*65}")
    print(f"  Total questions     : {scores['total_questions']}")
    print(f"  Retrieval Hit@k     : {scores['retrieval_hit_at_k']:.1%}")
    print(f"  Label Accuracy      : {scores['label_accuracy']:.1%}")
    if scores["refusal_accuracy"] is not None:
        print(f"  Refusal Accuracy    : {scores['refusal_accuracy']:.1%}")
    if scores["conflict_detection"] is not None:
        print(f"  Conflict Detection  : {scores['conflict_detection']:.1%}")
    print(f"\n  Per-trap breakdown:")
    print(f"  {'Trap':<8} {'N':>4} {'LabelAcc':>10} {'HitRate':>10}")
    print(f"  {'-'*36}")
    for trap, stats in scores["per_trap"].items():
        print(
            f"  {trap:<8} {stats['total']:>4} "
            f"{stats['label_acc']:>9.1%} "
            f"{stats['retrieval_hit']:>9.1%}"
        )
    print(f"{'='*65}\n")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the TimeMachine pipeline against a gold Q&A set."
    )
    parser.add_argument(
        "--split",
        choices=["dev", "held_out", "all"],
        default="dev",
        help="Which split to evaluate (default: dev).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to a specific gold JSONL file (overrides --split).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full pipeline trace for each question.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Save full results to a JSON file.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the embedding index from scratch.",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Call the LLM for natural language answer generation during eval (slower).",
    )
    args = parser.parse_args()

    # Resolve gold file
    root = Path(__file__).parent
    if args.file:
        gold_file = args.file
        split_name = gold_file.stem
    elif args.split == "dev":
        gold_file = root / "gold_qa_dev.jsonl"
        split_name = "dev"
    elif args.split == "held_out":
        gold_file = root / "gold_qa_heldout.jsonl"
        split_name = "held_out"
    else:
        gold_file = root / "gold_qa.jsonl"
        split_name = "all"

    if not gold_file.exists():
        print(f"[ERROR] Gold file not found: {gold_file}")
        sys.exit(1)

    # Initialise pipeline (one-time setup)
    pipeline = TimeMachinePipeline(rebuild_index=args.rebuild_index)

    # Run evaluation (default skip_llm=True unless --with-llm is specified)
    scores = run_evaluation(gold_file, pipeline, verbose=args.verbose, skip_llm=not args.with_llm)

    # Print report
    print_report(scores, split_name)

    # Optionally save results
    if args.out:
        out_path = args.out
        out_path.write_text(json.dumps(scores, indent=2, default=str))
        print(f"[Eval] Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
