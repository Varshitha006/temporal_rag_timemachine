"""
eval.py
--------
V3: Evaluation harness for the TimeMachine pipeline.

Scores the pipeline against gold_qa_dev.jsonl on FIVE metrics:
  1. Retrieval Hit@k         : Was the governing clause in the top-k retrieved?
  2. Label Accuracy          : Does the pipeline's label match the gold label?
  3. Refusal Accuracy        : Of 'refuse' questions, did the pipeline refuse?
  4. Conflict Detection      : Of 'conflict' questions, was a conflict detected?
  5. Citation Precision (NEW): fraction of cited clauses that are in the gold
                               governing set.  hit@k passes even if only ONE
                               gold clause is found, hiding citation noise;
                               citation-precision penalises extra wrong citations.

Also reports per-trap-type breakdown (T1–T5) and a summary table.

V3 additions:
  --no-expansion : run with use_graph_expansion=False for ablation comparison
  Prints a side-by-side comparison table when both runs complete.

Usage:
  python eval.py                         # eval dev set (expansion ON)
  python eval.py --no-expansion          # ablation: expansion OFF
  python eval.py --split held_out        # (end of project only)
  python eval.py --file gold_qa_dev.jsonl
  python eval.py --verbose
"""

import os
import sys
import subprocess
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

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
    appears in the pipeline's citations.  This is Hit@k.
    """
    if not governing:
        return True   # refuse/escalate questions: no governing clause needed
    return bool(set(governing) & set(citations))


def compute_citation_precision(governing: list[str], citations: list[str]) -> Optional[float]:
    """
    Citation Precision (NEW metric):
      |cited ∩ gold_governing| / |cited|

    Returns None when the gold label is 'refuse'/'escalate' (no governing set)
    or when there are no citations (avoid division by zero).

    Interpretation:
      1.0 = every cited clause is in the gold governing set (no noise)
      0.5 = half the cited clauses are gold, half are irrelevant
      0.0 = no cited clause is gold
    """
    if not governing:
        return None   # refuse/escalate: no meaningful gold set
    if not citations:
        return None   # no citations (e.g., refuse prediction): skip
    gold_set = set(governing)
    cited_set = set(citations)
    correct = len(gold_set & cited_set)
    return correct / len(cited_set)


def compute_label_match(gold_label: str, pred_label: str) -> bool:
    """
    Label accuracy: 'escalate' and 'conflict' are treated as equivalent
    (both surface ambiguity to the user).
    """
    NORM = {
        "escalate": "conflict",
        "answer": "answerable",
    }
    return NORM.get(gold_label, gold_label) == NORM.get(pred_label, pred_label)


def compute_refusal_accuracy(gold_label: str, pred_label: str) -> Optional[bool]:
    """Only relevant for gold_label='refuse'."""
    if gold_label != "refuse":
        return None
    return pred_label in {"refuse", "escalate"}


def compute_conflict_accuracy(gold_label: str, pred_label: str) -> Optional[bool]:
    """Only relevant for gold_label='conflict'."""
    if gold_label != "conflict":
        return None
    return pred_label == "conflict"


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
    retrieval_hits   = 0
    label_matches    = 0
    refusal_correct  = 0
    refusal_total    = 0
    conflict_correct = 0
    conflict_total   = 0
    citation_prec_sum   = 0.0
    citation_prec_count = 0

    # Per-trap breakdown
    trap_stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "label_match": 0, "hit": 0,
        "cit_prec_sum": 0.0, "cit_prec_count": 0,
    })

    per_question_results = []

    for item in gold_items:
        qid        = item["id"]
        question   = item["question"]
        as_of      = item.get("as_of_date")
        gold_label = item["label"]
        governing  = item.get("governing", [])
        trap_type  = item.get("trap", "none")

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
        cprec = compute_citation_precision(governing, result.citations)

        # Accumulate aggregate
        retrieval_hits += int(hit)
        label_matches  += int(lmatch)
        if ref is not None:
            refusal_total   += 1
            refusal_correct += int(ref)
        if con is not None:
            conflict_total   += 1
            conflict_correct += int(con)
        if cprec is not None:
            citation_prec_sum   += cprec
            citation_prec_count += 1

        # Per-trap accumulation
        trap_stats[trap_type]["total"]       += 1
        trap_stats[trap_type]["label_match"] += int(lmatch)
        trap_stats[trap_type]["hit"]         += int(hit)
        if cprec is not None:
            trap_stats[trap_type]["cit_prec_sum"]   += cprec
            trap_stats[trap_type]["cit_prec_count"] += 1

        # Store result for full reporting
        per_question_results.append({
            "id":             qid,
            "question":       question[:60],
            "gold_label":     gold_label,
            "pred_label":     result.label,
            "trap":           trap_type,
            "hit":            hit,
            "label_match":    lmatch,
            "citation_prec":  round(cprec, 3) if cprec is not None else None,
            "citations":      result.citations,
            "expanded_docs":  result.expanded_docs,
            "answer_snippet": result.answer[:120] if result.answer else "",
        })

        # Per-question summary line
        status  = "OK"  if lmatch else "ERR"
        hit_str = "HIT" if hit    else "MISS"
        cprec_str = f"P={cprec:.2f}" if cprec is not None else "P=N/A"
        print(
            f"  [{status:3s}] {qid} | gold={gold_label:9s} pred={result.label:9s} "
            f"trap={trap_type:4s} | {hit_str} {cprec_str}"
        )

    # ── Aggregate scores ───────────────────────────────────────────────────────
    scores = {
        "total_questions":    total,
        "retrieval_hit_at_k": retrieval_hits / total,
        "label_accuracy":     label_matches  / total,
        "refusal_accuracy":   refusal_correct / refusal_total if refusal_total > 0 else None,
        "conflict_detection": conflict_correct / conflict_total if conflict_total > 0 else None,
        "citation_precision": citation_prec_sum / citation_prec_count if citation_prec_count > 0 else None,
        "per_trap": {
            trap: {
                "total":          stats["total"],
                "label_acc":      stats["label_match"] / stats["total"],
                "retrieval_hit":  stats["hit"] / stats["total"],
                "citation_prec":  (
                    stats["cit_prec_sum"] / stats["cit_prec_count"]
                    if stats["cit_prec_count"] > 0 else None
                ),
            }
            for trap, stats in sorted(trap_stats.items())
        },
        "per_question": per_question_results,
    }

    return scores


# ── Report Printing ───────────────────────────────────────────────────────────

def print_report(scores: dict, split_name: str, expansion: bool = True):
    """Pretty-print the evaluation report."""
    flag = "expansion=ON" if expansion else "expansion=OFF"
    print(f"\n{'='*70}")
    print(f"  TimeMachine Evaluation Report — {split_name.upper()} SET  [{flag}]")
    print(f"{'='*70}")
    print(f"  Total questions        : {scores['total_questions']}")
    print(f"  Retrieval Hit@k        : {scores['retrieval_hit_at_k']:.1%}")
    print(f"  Label Accuracy         : {scores['label_accuracy']:.1%}")
    if scores["refusal_accuracy"] is not None:
        print(f"  Refusal Accuracy       : {scores['refusal_accuracy']:.1%}")
    if scores["conflict_detection"] is not None:
        print(f"  Conflict Detection     : {scores['conflict_detection']:.1%}")
    if scores["citation_precision"] is not None:
        print(f"  Citation Precision     : {scores['citation_precision']:.1%}  (<-- NEW)")
    print(f"\n  Per-trap breakdown:")
    print(f"  {'Trap':<8} {'N':>4} {'LabelAcc':>10} {'HitRate':>10} {'CitPrec':>10}")
    # (CitPrec = citation precision: cited-in-gold / total-cited)
    print(f"  {'-'*46}")
    for trap, stats in scores["per_trap"].items():
        cprec = stats.get("citation_prec")
        cprec_str = f"{cprec:9.1%}" if cprec is not None else "       N/A"
        print(
            f"  {trap:<8} {stats['total']:>4} "
            f"{stats['label_acc']:>9.1%} "
            f"{stats['retrieval_hit']:>9.1%}"
            f"{cprec_str}"
        )
    print(f"{'='*70}\n")


def print_comparison_table(scores_on: dict, scores_off: dict):
    """
    Print a side-by-side comparison of expansion=ON vs expansion=OFF.
    This is the ablation table requested in the acceptance criteria.
    """
    print(f"\n{'='*80}")
    print(f"  ABLATION: expansion=ON  vs  expansion=OFF")
    print(f"{'='*80}")
    metrics = [
        ("Retrieval Hit@k",    "retrieval_hit_at_k"),
        ("Label Accuracy",     "label_accuracy"),
        ("Refusal Accuracy",   "refusal_accuracy"),
        ("Conflict Detection", "conflict_detection"),
        ("Citation Precision", "citation_precision"),
    ]
    print(f"  {'Metric':<25} {'expansion=ON':>15} {'expansion=OFF':>15} {'Delta':>10}")
    print(f"  {'-'*65}")
    for label, key in metrics:
        v_on  = scores_on.get(key)
        v_off = scores_off.get(key)
        if v_on is None or v_off is None:
            print(f"  {label:<25} {'N/A':>15} {'N/A':>15} {'N/A':>10}")
        else:
            delta = v_on - v_off
            sign  = "+" if delta >= 0 else ""
            print(
                f"  {label:<25} {v_on:>14.1%} {v_off:>14.1%} {sign}{delta:>9.1%}"
            )
    print(f"\n  Per-trap comparison (Label Accuracy):")
    print(f"  {'Trap':<8} {'ON':>10} {'OFF':>10} {'Delta':>10}")
    print(f"  {'-'*42}")
    all_traps = sorted(set(scores_on["per_trap"].keys()) | set(scores_off["per_trap"].keys()))
    for trap in all_traps:
        on_acc  = scores_on["per_trap"].get(trap, {}).get("label_acc")
        off_acc = scores_off["per_trap"].get(trap, {}).get("label_acc")
        if on_acc is None or off_acc is None:
            print(f"  {trap:<8} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            delta = on_acc - off_acc
            sign  = "+" if delta >= 0 else ""
            print(f"  {trap:<8} {on_acc:>9.1%} {off_acc:>9.1%} {sign}{delta:>9.1%}")
    print(f"{'='*80}\n")


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
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        help="Disable KG graph expansion (ablation: compare with and without expansion).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run BOTH expansion=ON and expansion=OFF and print a comparison table.",
    )
    args = parser.parse_args()

    # Resolve gold file
    root = Path(__file__).parent
    if args.file:
        gold_file  = args.file
        split_name = gold_file.stem
    elif args.split == "dev":
        gold_file  = root / "gold_qa_dev.jsonl"
        split_name = "dev"
    elif args.split == "held_out":
        gold_file  = root / "gold_qa_heldout.jsonl"
        split_name = "held_out"
    else:
        gold_file  = root / "gold_qa.jsonl"
        split_name = "all"

    if not gold_file.exists():
        print(f"[ERROR] Gold file not found: {gold_file}")
        sys.exit(1)

    skip_llm = not args.with_llm

    if args.compare:
        # Run both ON and OFF and print comparison
        print("[Eval] Running expansion=ON ...")
        pipeline_on = TimeMachinePipeline(
            rebuild_index=args.rebuild_index,
            use_graph_expansion=True,
        )
        scores_on = run_evaluation(gold_file, pipeline_on, verbose=args.verbose, skip_llm=skip_llm)
        print_report(scores_on, split_name, expansion=True)

        print("[Eval] Running expansion=OFF (ablation) ...")
        # Reuse same pipeline but disable expansion flag (cheaper than rebuilding index)
        pipeline_on.use_graph_expansion = False
        scores_off = run_evaluation(gold_file, pipeline_on, verbose=False, skip_llm=skip_llm)
        print_report(scores_off, split_name, expansion=False)

        print_comparison_table(scores_on, scores_off)

        if args.out:
            combined = {"expansion_on": scores_on, "expansion_off": scores_off}
            args.out.write_text(json.dumps(combined, indent=2, default=str))
            print(f"[Eval] Full results saved to: {args.out}")

    else:
        # Single run
        use_expansion = not args.no_expansion
        pipeline = TimeMachinePipeline(
            rebuild_index=args.rebuild_index,
            use_graph_expansion=use_expansion,
        )
        scores = run_evaluation(gold_file, pipeline, verbose=args.verbose, skip_llm=skip_llm)
        print_report(scores, split_name, expansion=use_expansion)

        if args.out:
            args.out.write_text(json.dumps(scores, indent=2, default=str))
            print(f"[Eval] Full results saved to: {args.out}")


if __name__ == "__main__":
    main()
