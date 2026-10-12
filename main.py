"""
main.py
--------
Interactive CLI for the TimeMachine pipeline.

Lets you ask questions interactively or run a single question from the command line.

Usage:
  python main.py                                            # interactive mode
  python main.py --question "What is the WFH policy?"     # single question
  python main.py --question "..." --date 2024-03-01        # with as_of_date
  python main.py --rebuild                                  # rebuild embedding index

Example interactive session:
  > What is the WFH policy?
  > What is the password policy? [date=2024-03-01]
  > Can a contractor work from home? [date=2024-02-01]
  > quit
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Auto-detect and re-execute with project .venv python if available and not yet active
venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and os.environ.get("TIMEMACHINE_VENV_ACTIVE") != "1":
    # Prevent infinite loop by setting an environment flag
    os.environ["TIMEMACHINE_VENV_ACTIVE"] = "1"
    try:
        sys.exit(subprocess.call([str(venv_python)] + sys.argv))
    except Exception:
        pass   # Fallback to current python if subprocess fails

from src.pipeline import TimeMachinePipeline


def print_result(result):
    """Pretty-print a PipelineResult to the terminal using ASCII formatting."""
    print(f"\n{'-'*60}")
    print(f"  QUESTION  : {result.question}")
    print(f"  AS-OF DATE: {result.as_of_date}")
    print(f"  DECISION  : {result.label.upper()}")
    print(f"  CITATIONS : {', '.join(result.citations) if result.citations else '(none)'}")
    print(f"{'-'*60}")
    print(f"\n{result.answer}\n")
    print(f"  [Stats] Retrieved {result.retrieved_count} clauses -> {result.in_force_count} in force")
    print()


def interactive_loop(pipeline: TimeMachinePipeline):
    """Run an interactive question-answer session."""
    print("\n" + "="*60)
    print("  TimeMachine — Temporal Policy Compliance Copilot")
    print("  Corpus: Solstice Robotics Pvt Ltd (44 policy documents)")
    print("="*60)
    print("  Type your question. Optionally add [date=YYYY-MM-DD].")
    print("  Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exiting]")
            break

        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            print("[Bye!]")
            break

        # Parse optional [date=YYYY-MM-DD] suffix
        as_of_date = None
        question = raw
        if "[date=" in raw:
            parts = raw.split("[date=")
            question = parts[0].strip()
            date_part = parts[1].rstrip("]").strip()
            as_of_date = date_part

        # Run pipeline
        result = pipeline.query(question, as_of_date=as_of_date)
        print_result(result)


def main():
    parser = argparse.ArgumentParser(
        description="TimeMachine — Temporal Policy Compliance Copilot"
    )
    parser.add_argument(
        "--question", "--query", "-q",
        type=str,
        default=None,
        help="Single question to answer (non-interactive mode).",
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=None,
        help="as_of_date for the query (ISO format: YYYY-MM-DD). Default: 2025-06-01.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the embedding index from scratch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full pipeline trace (retrieval, filtering, decision log).",
    )
    args = parser.parse_args()

    # Initialise the pipeline
    pipeline = TimeMachinePipeline(rebuild_index=args.rebuild)

    if args.question:
        # Single-question mode
        result = pipeline.query(
            question=args.question,
            as_of_date=args.date,
            verbose=args.verbose,
        )
        print_result(result)
    else:
        # Interactive mode
        interactive_loop(pipeline)


if __name__ == "__main__":
    main()
