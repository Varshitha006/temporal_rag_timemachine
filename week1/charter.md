# M0 Charter — TimeMachine: A Temporal Policy-Compliance Copilot
**Team:** [Fill in team name / roster]
**Problem:** Problem A — TimeMachine
**Date:** Week 1

---

## 1. Domain and corpus source

We chose a **fictional company handbook** ("Solstice Robotics Pvt Ltd") over a real regulator's
circulars for three reasons:

1. **Control over trap density.** The problem statement requires five specific trap types
   (T1–T5), each with genuine contradictions, supersessions, and silences. Authoring a fictional
   corpus lets us guarantee every trap type is present and precisely dated, rather than hoping a
   real dataset happens to contain enough of them.
2. **No leakage risk.** A real company handbook or regulator circular can sometimes be found
   online and could be seen (in part) by a foundation model at pretraining time. A fully
   invented company, invented policy numbers, and invented dates cannot be recalled from an
   LLM's memory — the model is forced to actually retrieve from our corpus rather than
   pattern-match on prior knowledge.
3. **Realistic structure without real-world noise.** Real circulars are often scanned PDFs with
   inconsistent formatting, which would turn this into a document-parsing project instead of a
   temporal-reasoning project. Our corpus keeps clean, clause-numbered markdown so the challenge
   stays where the rubric wants it: retrieval, temporal grounding, conflict detection, and
   refusal — not OCR.

**Domain:** internal corporate policy (HR, IT Security, Finance/Procurement, Legal &
Compliance) — 15 policy families, 44 documents, spanning **2019-01-01 to 2025 (current)**, i.e.
more than four years as required, with 14 explicit supersession links (well above the minimum
of three).

## 2. Metadata schema

Every document carries this metadata (YAML front-matter in the `.md` file, mirrored in
`metadata.csv` for fast programmatic loading):

| Field | Type | Purpose |
|---|---|---|
| `doc_id` | string (`D001`–`D044`) | Stable identifier, referenced by clause id (`D003:C1`) |
| `title` | string | Human-readable policy name |
| `issuing_authority` | string | Who issued it — needed for the "authority" leg of precedence |
| `effective_date` | ISO date | Start of validity |
| `sunset_date` | ISO date or null | End of validity (null = still current) |
| `supersedes` | doc_id or null | Direct predecessor in the same policy family |
| `version` | string | Human version label (v1.0, v2.0, ...) |
| `category` | string | Policy family, used to group versions and detect same-topic conflicts |

We deliberately **did not** add a numeric "priority" field. Precedence is *derived*, not stored
directly — see Section 3 — because in the real world a document's authority (e.g., Legal &
Compliance vs Regional HR) is not always encoded on the document itself, and we want the
system's reasoning to be inspectable and justified, not a lookup into a hidden ranking.

## 3. Precedence rules we will adopt (recency vs specificity vs authority)

We use a **three-tier precedence order**, applied in this sequence when two in-force documents
disagree on the same question:

1. **Explicit override / supersession** (highest). If document B's `supersedes` field points to
   document A, or B's text explicitly states it overrides A (as our GDPR Addendum and Emergency
   Procurement Circular both do), B wins outright — this is not really a "conflict," it's a
   stated relationship, so it should be resolved automatically without flagging ambiguity to the
   user.
2. **Specificity.** Absent an explicit override, a document that is narrower in scope (e.g.,
   "EU clients only," "Bengaluru office only," "disaster-recovery period only") wins over a
   general document on points within its narrower scope, and the general document continues to
   govern everything outside that scope. This is standard legal-drafting practice (*lex
   specialis*) and matches how our GDPR Addendum (D011) and the Bengaluru Circular (D004) are
   written.
3. **Recency, gated by effective/sunset windows.** Within the same specificity tier, the
   document whose `[effective_date, sunset_date)` window contains the query's `as_of_date` wins.
   If two documents of equal specificity are *both* in force on the same date and neither
   states an override (this should not happen in a well-curated corpus, but the detector must
   catch it if it does), the system reports a **genuine contradiction** (label: `conflict`) and
   does not silently guess — this is the T2 trap category.

**Issuing authority is used only as a tie-breaker signal in the decision log / explanation**,
not as an automatic ranking rule — we found that a rule like "Legal always outranks HR" breaks
down quickly (e.g., a Legal circular addressing something clearly inside HR's remit), so we
treat authority as a factor a human reviewer should be able to see, not something the system
resolves silently on the user's behalf.

## 4. Framework choice

We will use **LangGraph** for the agent/orchestration layer.

| Option | Why we didn't choose it |
|---|---|
| Google ADK | Strong for tool-calling but less mature ecosystem for explicit graph-based state machines, which we need for the retrieve → temporal-filter → conflict-check → decision-gate pipeline. |
| OpenAI Agents SDK | Good DX, but ties us more tightly to OpenAI-hosted models; we want to freely swap between a local model (Ollama) and a free-tier hosted model without re-architecting. |
| CrewAI | Optimized for role-based multi-agent collaboration (better fit for Problem B / IncidentOps); our pipeline is a single linear-with-branches reasoning flow, not a crew of specialized agents debating. |
| AutoGen | Similar reasoning — better suited to open-ended multi-agent conversation than to a deterministic, auditable decision pipeline. |

**Why LangGraph fits us:** our architecture (Section A.6 of the brief) is explicitly a directed
graph with conditional branches (`answer / clarify / refuse / escalate`), not a free-form agent
loop. LangGraph lets us define each stage (retriever, temporal filter, conflict service,
decision gate) as an explicit node with typed state passed between them, which makes the
**decision log** and the **held-out evaluation** far easier to reproduce deterministically —
important given the rubric weights honest, reproducible evaluation heavily (Section A.8, #4).

## 5. What "good" means for this project

- **Faithfulness**: every non-refused answer must cite a specific `doc_id:clause_id` that
  actually supports the claim made.
- **Calibrated refusal**: T4-type questions (genuinely unanswerable) must be refused, not
  answered by analogy to an adjacent policy.
- **Temporal correctness**: for any question with an `as_of_date`, the answer must reflect the
  document(s) in force on that exact date, including at the exact boundary date.
- **Conflict transparency**: when two in-force documents disagree, the system must say so and
  show its precedence reasoning, not silently pick one.

These four criteria map directly onto the eval harness we will build in M2 (`eval.py`):
retrieval hit@k, answer faithfulness, refusal accuracy, and (new, added by us) a
**conflict-detection accuracy** metric scored per-trap using the ledger in `traps.md`.
