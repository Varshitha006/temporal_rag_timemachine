# Week 1 — M0: Framing & Data Plan

**Module:** M0 (per problem statement, Section A.5)
**Goal:** Agree the domain, corpus source, and what "good" means, before writing any retrieval
or agent code.

## What's in this folder

| File | What it is |
|---|---|
| `charter.md` | The 2-page M0 charter: corpus source and rationale, metadata schema, precedence rules, framework choice + justification, and our operational definition of "good" (faithfulness / calibrated refusal / temporal correctness / conflict transparency). |
| `week1_README.md` | This file. |

## Summary of decisions made this week

1. **Corpus domain chosen:** internal corporate policy for a fictional company, "Solstice
   Robotics Pvt Ltd," covering 15 policy families (WFH, Leave, Data Retention, InfoSec/Password,
   Travel & Expense, Procurement, Code of Conduct, BYOD, Contractor Engagement, Exit, Equipment
   Allowance, Social Media, Whistleblower, Data Classification, Anti-Bribery, Parental Leave,
   Background Verification).
   - **Why fictional, not a real regulator's circulars:** guarantees we can plant every required
     trap type precisely, and guarantees no LLM has seen this data at pretraining time (a real,
     Google-indexed policy document could theoretically leak into a foundation model's training
     data; an invented company with invented doc IDs and dates cannot).

2. **Metadata schema locked in:** `doc_id, title, issuing_authority, effective_date,
   sunset_date, supersedes, version, category`. This directly supports the temporal filter
   (M3) and the conflict/precedence detector (M4) without needing a redesign later.

3. **Precedence policy decided up front** (see charter §3): explicit override beats
   specificity beats recency-within-window; true conflicts (same specificity, same window, no
   stated override) are surfaced to the user rather than silently resolved. This decision was
   made *before* writing retrieval code so that M2's baseline can be evaluated against a
   pre-committed definition of "correct," not one invented after seeing what the baseline gets
   wrong.

4. **Framework selected:** LangGraph, chosen because our pipeline
   (retrieve → temporal-filter → conflict-check → decision-gate) is a deterministic branching
   graph, not an open-ended multi-agent conversation — see charter §4 for the full comparison
   against ADK, OpenAI Agents SDK, CrewAI, and AutoGen.

5. **Definition of "good" fixed for the whole project:** faithfulness (every answer cites a
   real clause), calibrated refusal (silent/missing → refuse, don't guess), temporal
   correctness (right document for the right date, including boundary dates), and conflict
   transparency (say so when two in-force documents disagree). These four become the four
   scored dimensions of `eval.py` in Week 3–4 (M2).

## Why this matters for later weeks

Every module from here on is graded against the commitments made this week:
- M2's evaluation harness computes exactly the four metrics defined in Charter §5.
- M3's temporal filter implements exactly the precedence order defined in Charter §3.
- M4's conflict detector implements the same precedence order to decide `answer` vs
  `conflict` vs `escalate` vs `refuse`.

Locking these decisions in Week 1 — before any code or corpus trap was written — is itself
part of the decision log: it shows the definitions were not backfilled to match whatever the
system happened to produce later.

## Checkpoint status

☐ Charter reviewed with instructor (schedule this before moving to M1 corpus curation).
