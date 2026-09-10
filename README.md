# temporal_rag_timemachine
A temporal RAG system that answers policy questions as of a point in time — with clause-level citations, conflict detection, and calibrated refusal.

# TimeMachine — Temporal Policy-Compliance Copilot
INT450 Capstone, Problem A. Weeks 1–2 deliverables (M0 + M1).

## Folder guide

```
TimeMachine_Project/
├── week1_M0_framing/
│   ├── charter.md          ← 2-page M0 charter (corpus source, metadata schema,
│   │                          precedence rules, framework choice + justification)
│   └── week1_README.md     ← Week 1 summary: what was decided and why
│
└── week2_M1_corpus/
    ├── corpus/
    │   ├── D001.md ... D044.md   ← 44 policy documents (the private dataset)
    │   └── metadata.csv          ← machine-readable metadata for all 44 docs
    ├── traps.md                  ← ledger of every planted trap (T1–T5) and where it lives
    ├── gold_qa.jsonl             ← all 80 gold Q&A pairs, with dev/held_out split field
    ├── gold_qa_dev.jsonl         ← 56 dev questions (use freely during development)
    ├── gold_qa_heldout.jsonl     ← 24 held-out questions (seal until Week 8 / M5)
    └── week2_README.md           ← Week 2 summary: corpus design, trap coverage, QA design
```

## How to read this in order

1. Start with `week1_M0_framing/week1_README.md` for the framing decisions.
2. Read `week1_M0_framing/charter.md` for the full reasoning behind those decisions.
3. Read `week2_M1_corpus/week2_README.md` for how the corpus and traps were built.
4. Skim a few `corpus/D0xx.md` files and `traps.md` side by side to see how a trap is actually
   implemented in the text.
5. Open `gold_qa.jsonl` (or `gold_qa_dev.jsonl`) to see how questions map to expected answers
   and governing clauses.

## What's next (Week 3 onward — not part of this deliverable)

- **M2** (Weeks 3–4): baseline hybrid retrieval + `eval.py` scoring hit@k, faithfulness, and
  refusal accuracy on `gold_qa_dev.jsonl`. Expect it to fail most T1/T3 questions — that's the
  documented, intended failure that motivates M3.
- **M3** (Weeks 4–5): temporal filter using `effective_date`/`sunset_date` from `metadata.csv`.
- **M4** (Weeks 6–7): knowledge graph over `supersedes` links + conflict detector implementing
  the precedence rules from `charter.md` §3 + the decision gate (answer/clarify/refuse/escalate).
- **M5** (Week 8): run once against `gold_qa_heldout.jsonl`, write the evaluation report,
  decision log, and AI-use appendix.
