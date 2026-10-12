# TimeMachine — Temporal Policy-Compliance Copilot

**Project:** TimeMachine (Problem A)  
**Company (fictional):** Solstice Robotics Pvt Ltd  
**Documents:** 44 policy documents · 15 families · 2019–2025  
**Gold Q&A:** 80 questions (56 dev / 24 held-out)

---

## What this project does

TimeMachine answers questions about internal company policy *with temporal awareness*:
- It knows **which version of a policy was in force on a specific date**
- It detects **contradictions between simultaneously valid documents**
- It **refuses to answer** when the corpus is genuinely silent (rather than hallucinating)
- It **escalates** when a question's scope is ambiguous
- Every answer is **cited** with a specific `doc_id:clause_id`

---

## Architecture

```
User Question + as_of_date
        │
        ▼
┌───────────────────┐
│  1. Retriever     │  Semantic search over all 128 clauses
│  (all-MiniLM-L6)  │  Returns top-k by cosine similarity
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 2. Temporal Filter│  Filters to clauses in-force on as_of_date
│                   │  Annotates with supersession / expiry notes
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 3. Knowledge Graph│  NetworkX graph of supersession chains,
│   (NetworkX)      │  same-category links, conflict-risk edges
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 4. Conflict Det.  │  Three-tier precedence (charter.md §3):
│   + Decision Gate │    Tier 1: Explicit supersession
│                   │    Tier 2: Specificity (lex specialis)
│                   │    Tier 3: Recency
│                   │  → answer | conflict | escalate | refuse
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  5. Gemini LLM    │  Generates natural-language answer
│  (google-genai)   │  citing specific clause IDs
└─────────┬─────────┘
          │
          ▼
   Answer + Citations + Decision Log
```

---

## Folder Structure

```
temporal_rag_timemachine/
├── corpus/                   # 44 policy documents + metadata
│   ├── D001.md ... D044.md   #   Policy docs (YAML front-matter + clauses)
│   └── metadata.csv          #   Authoritative metadata for all docs
│
├── src/                      # Core pipeline modules
│   ├── __init__.py
│   ├── corpus_loader.py      #   Loads corpus -> DocMeta + Clause objects
│   ├── knowledge_graph.py    #   Builds NetworkX KG of policy relationships
│   ├── retriever.py          #   Semantic retrieval (sentence-transformers)
│   ├── temporal_filter.py    #   Temporal filtering + precedence annotation
│   ├── conflict_detector.py  #   Conflict detection + decision gate
│   ├── llm_client.py         #   Gemini API client + prompt templates
│   └── pipeline.py           #   Full end-to-end orchestrator
│
├── docs/                     # Project planning and documentation
│   ├── charter.md            #   M0 charter: design decisions
│   ├── traps.md              #   Trap ledger: T1-T5 planted traps
│   ├── week1_README.md       #   Week 1 framing notes
│   └── week2_README.md       #   Week 2 corpus curation notes
│
├── index/                    # Auto-generated (created on first run)
│   ├── embeddings.npy        #   Cached clause embeddings
│   └── clause_ids.json       #   Clause ID order for the embedding matrix
│
├── eval.py                   # Evaluation harness (4 metrics + per-trap breakdown)
├── main.py                   # Interactive CLI
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed to git)
│
├── gold_qa.jsonl             # All 80 gold Q&A pairs
├── gold_qa_dev.jsonl         # 56 dev questions (70%)
└── gold_qa_heldout.jsonl     # 24 held-out questions (30%) — sealed until final eval
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A Google AI API key (get one at [aistudio.google.com](https://aistudio.google.com))

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or if using the project's virtual environment:
```bash
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 3. Configure environment

Edit `.env` to set your API key:

```env
LLM_API_KEY=your_google_api_key_here
LLM_MODEL=gemini-1.5-flash
```

---

## Usage

### Interactive mode

```bash
python main.py
```

Then type questions like:
```
You: What is the WFH policy?
You: Is MFA mandatory? [date=2024-03-01]
You: How many sick leave days do employees get? [date=2025-06-01]
You: quit
```

### Single question mode

```bash
python main.py --question "What is the WFH policy?" --date 2024-03-01
```

### Run evaluation (dev set)

```bash
python eval.py
```

### Run evaluation with full trace

```bash
python eval.py --verbose
```

### Run on held-out set (final evaluation only)

```bash
python eval.py --split held_out
```

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **Retrieval Hit@k** | Is at least one governing clause in the top-k retrieved? |
| **Label Accuracy** | Does the pipeline's output label match the gold label? |
| **Refusal Accuracy** | For `refuse` questions: did the pipeline refuse (not hallucinate)? |
| **Conflict Detection** | For `conflict` questions: was a conflict detected? |

Per-trap-type breakdown (T1–T5 and control) is also reported.

---

## Trap Types (from `traps.md`)

| Trap | What it tests | Count |
|------|--------------|-------|
| **T1** Superseded-but-similar | Does retrieval blindly follow similarity to old docs? | 5 |
| **T2** Contradiction | Are simultaneous conflicts detected and resolved via precedence? | 6 |
| **T3** Temporal edge | Does the answer correctly flip at a boundary date? | 8 |
| **T4** Silent/missing | Does the system refuse rather than hallucinate? | 8 |
| **T5** Ambiguous scope | Does the system escalate rather than over-apply a rule? | 8 |

---

## Design Decisions

### Why RAG + Knowledge Graph?

Pure semantic RAG (retrieval + LLM) fails on this corpus because:
1. **T1 traps**: Old documents have near-identical phrasing to current ones — semantic similarity alone picks the wrong version.
2. **T2/T3 traps**: The answer depends on *when* a policy is in force, not just what it says.
3. **T4/T5 traps**: The system must know when to *not* answer — something LLMs struggle with without structure.

The KG adds:
- **Supersession chains**: explicit `A → B` edges so the filter can find the canonical current version.
- **Conflict-risk edges**: pairs of docs that share a category and overlapping windows.
- **Authority tiers**: used as tie-breaker signal to flag non-owning-authority overrides (T2-d, T2-f).

### Why local embeddings (sentence-transformers)?

- No API cost for retrieval (all 128 clauses embedded once, ~2 seconds).
- Embeddings cached to `index/` after first run.
- `all-MiniLM-L6-v2` is well-suited for short policy-clause texts.

### Why Gemini for the final LLM step?

- The LLM is called **only once per query**, only at the final answer-generation step.
- The structured `DecisionResult` from the pipeline constrains what the LLM can say.
- This prevents hallucination: the LLM only sees the clauses the pipeline deemed in-force.

### Precedence rules (from `charter.md §3`)

1. **Explicit supersession** (highest): `supersedes` pointer → predecessor loses.
2. **Specificity**: time-boxed or narrower-scope document wins within its scope (*lex specialis*), **unless** the issuing authority doesn't own the policy category (T2-d, T2-f → escalate).
3. **Recency**: more recent effective date wins within same specificity tier.

---

## Module Reference

| Module | Key class/function | Purpose |
|--------|-------------------|---------|
| `src/corpus_loader.py` | `load_corpus()` | Load 44 docs → DocMeta + Clause objects |
| `src/knowledge_graph.py` | `build_knowledge_graph()` | Build NetworkX KG |
| `src/retriever.py` | `Retriever.search()` | Semantic clause retrieval |
| `src/temporal_filter.py` | `temporal_filter()`, `apply_precedence()` | Date-based filtering + annotation |
| `src/conflict_detector.py` | `decide()` | Three-tier precedence + decision gate |
| `src/llm_client.py` | `LLMClient.generate()` | Gemini final answer generation |
| `src/pipeline.py` | `TimeMachinePipeline.query()` | Full end-to-end orchestrator |
| `eval.py` | `run_evaluation()` | 4-metric eval harness |
| `main.py` | `interactive_loop()` | Interactive CLI |

---

## Key Files Not to Modify

- `gold_qa_heldout.jsonl` — sealed until the final M5 evaluation
- `corpus/*.md` — the corpus is fixed; changes would invalidate the gold set
- `corpus/metadata.csv` — authoritative source of temporal metadata
