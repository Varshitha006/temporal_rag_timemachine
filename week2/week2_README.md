# Week 2 — M1: Corpus Curation & Trap Injection

**Module:** M1 (per problem statement, Section A.5)
**Goal:** Build the private corpus, its metadata, the planted traps, and the gold Q&A set —
the backbone of the whole project's integrity model (Section 0.3 of the problem statement).

## What's in this folder

```
week2_M1_corpus/
├── corpus/
│   ├── D001.md ... D044.md      # 44 policy documents (markdown, YAML front-matter + clauses)
│   └── metadata.csv             # doc_id, title, authority, effective/sunset dates, supersedes, version, category
├── traps.md                     # Trap ledger: where every T1–T5 trap lives and which gold Qs test it
├── gold_qa.jsonl                # All 80 gold questions, with a "split" field (dev / held_out)
├── gold_qa_dev.jsonl            # The 56 dev questions (70%) — usable during development
├── gold_qa_heldout.jsonl        # The 24 held-out questions (30%) — DO NOT inspect during development
└── week2_README.md              # This file
```

**Important integrity note (per Section 0.3 of the problem statement):** `gold_qa_heldout.jsonl`
is meant to be used only once, at the very end (M5, Week 8), to report honest held-out numbers.
Looking at it now to "tune" the system defeats the purpose of the held-out split — treat it as
sealed until Week 8.

## 1. The corpus: what we built and why

**Fictional company:** Solstice Robotics Pvt Ltd. **44 documents, 15 policy families, spanning
2019-01-01 to 2025 (current)** — more than the 4-year minimum window required.

**14 explicit supersession chains** (well above the "at least three" minimum), so temporal
reasoning has real work to do, not just a token example:

| Family | Chain |
|---|---|
| WFH | D001 → D002 → D003 (+ D004 standalone regional exception) |
| Leave | D005 → D006 → D007 (+ D008 standalone time-boxed circular) |
| Data Retention | D009 → D010 (+ D011 standalone specific-scope addendum) |
| Password/InfoSec | D012 → D013 → D014 (+ D044 standalone field-ops exemption circular) |
| Travel & Expense | D015 → D016 (+ D017 standalone international addendum, D041 standalone client-site addendum) |
| Procurement | D018 → D019 (+ D020 standalone time-boxed emergency circular) |
| Code of Conduct | D021 → D022 |
| BYOD | D023 → D024 |
| Exit & Offboarding | D027 → D028 |
| Equipment Allowance | D029 → D030 → D031 |
| Gifts & Anti-Bribery | D035 → D036 (+ D043 standalone time-boxed regional circular) |
| Parental Leave | D037 → D038 |

Seven more families (Contractor Engagement, Social Media, Whistleblower, Data Classification,
Sales Enablement, Background Verification, and the Company Values decoy) are single-version
documents that exist to give the corpus realistic breadth and to support the T4/T5 traps — a
real handbook has plenty of policies that never change, plenty of narrow business-team
circulars that sit alongside the standing policies, and plenty of areas it never covers at all.

**Why 44, not more:** the problem statement is explicit that this is "a reasoning problem, not
a scale problem" (A.3). 44 well-structured, clause-numbered documents with real temporal
structure — including several standing policies deliberately paired with a narrower circular
issued by a different team — exercise the retrieval + temporal + conflict pipeline far more
thoroughly than 200 generic documents would.

**Document structure:** every file has YAML front-matter (`doc_id, title,
issuing_authority, effective_date, sunset_date, supersedes, version, category`) followed by
numbered clauses (`D0xx:C1`, `D0xx:C2`, ...), so the eventual retriever and citation module can
address a single clause precisely, and faithfulness can be checked against an exact string.

## 2. The trap taxonomy — what we planted and where

All five required trap types are planted, each with multiple documented instances and multiple
gold questions. Full detail with doc-level pointers is in `traps.md`; summary:

| Trap | What it tests | # instances | Example |
|---|---|---|---|
| **T1** Superseded-but-similar | Does retrieval blindly follow similarity? | 5 | Generic "WFH policy" query semantically nearest to the *retired* v1.0 |
| **T2** Contradiction | Conflict detection + precedence | 6 | GDPR Addendum (1-year EU retention) vs general policy (3-year) |
| **T3** Temporal edge | Does the answer correctly flip at a boundary date? | 8 | MFA becomes mandatory exactly on 2024-07-01 |
| **T4** Silent/missing | Calibrated refusal (no hallucination) | 8 | No document anywhere addresses sabbaticals |
| **T5** Ambiguous scope | Clarify rather than over-apply a rule | 8 | Can *this* contractor WFH — depends on their individual contract, which isn't in the corpus |

We also planted one **decoy document** (D040, Company Values & Mission Statement) that is
topically adjacent to almost every HR question but explicitly states it must not be cited as
governing policy — this checks that the system doesn't paper over a real T4 refusal by citing a
vague, non-binding statement instead.

**A deliberate design choice inside T2:** not every planted contradiction is meant to resolve
cleanly. Four of the six T2 instances (D010/D011, D019/D020, D017/D041, D036/D043) resolve
correctly by applying the precedence rules in `charter.md` §3 — an explicit override, or a
narrower document from the *same or a higher-authority* source. The other two (D034/D042 and
D014/D044) are constructed so that the narrower document comes from a business/operations team
that does *not* own the general policy it's trying to override (Sales & Marketing attempting to
waive a Data Classification NDA requirement; Field Operations attempting to waive an
IT-Security "no exceptions" MFA mandate). These two are intentionally **not** meant to
auto-resolve by specificity — the correct behaviour is to flag the conflict and escalate to the
policy's actual owner. This tests whether the conflict detector has learned the *reasoning*
behind precedence, or just the shortcut "narrower document always wins."

The same "don't just apply one shortcut" principle shows up in T5: the eight ambiguous-scope
questions span several different undefined-population patterns (contractor eligibility,
part-time vs full-time, intern protections, probation status, and a cross-regional role
boundary) rather than repeating one scenario with the nouns swapped, so a system that
generalizes the underlying pattern — "silence about scope is not itself an answer" — is
rewarded over one that has simply memorized a single rule about contractors.

## 3. The gold Q&A set

**80 questions total** (exceeds the ≥60 minimum with margin), each with:
`id, question, as_of_date (nullable), expected_answer, governing (list of doc_id:clause_id),
label, trap, split`.

**Label distribution:**

| Label | Count | Meaning |
|---|---|---|
| `answerable` | 57 | Has one correct, citable answer for the given `as_of_date` |
| `refuse` | 9 | Corpus is genuinely silent — correct behaviour is to decline |
| `escalate` | 8 | Scope is ambiguous — correct behaviour is to ask for clarification |
| `conflict` | 6 | Two in-force documents disagree — correct behaviour is to surface both and explain the precedence resolution (or, in two cases, to escalate rather than resolve automatically — see Section 2) |

**Trap coverage in the gold set:** T1 ×5, T2 ×6, T3 ×8, T4 ×8, T5 ×8, plus 45 "no trap"
control questions that establish a baseline of straightforward as-of lookups across every
policy family — without these controls, a low score on trap questions alone can't be
distinguished from a system that's just bad at basic retrieval.

**On the "as_of_date omitted" cases (Q004, Q005, Q058):** three questions deliberately omit
`as_of_date` to test the M3 decision documented in the charter — "what to do when as_of is
omitted." We fixed a reproducible evaluation reference date of **2025-06-01** for these; the
expected answers spell out both the correct current answer *and* the specific superseded
document a naive similarity-only retriever would incorrectly surface instead.

**70/30 split:** 56 dev questions / 24 held-out questions (exact 70/30), with every trap type
and every label represented in *both* splits, so a system can't score well on held-out purely
by luck of which trap types happened to land there.

## 4. Design decisions worth defending at the viva

1. **Why a fictional corpus instead of real circulars?** No leakage risk (an LLM cannot have
   memorized invented doc IDs/dates), and full control over trap density and dating — see
   `charter.md` §1 for the full reasoning.
2. **Why is "silent-at-that-date" treated as a T4 variant (Q043, Q051, Q056) rather than a
   temporal edge (T3)?** Because the correct behaviour is refusal (there is no rule to apply
   yet), not a flip between two competing rules — this distinction matters because our decision
   gate (M4) has different logic paths for "answer changed" vs "no answer exists yet," and the
   gold labels need to force that distinction to be tested.
3. **Why does the Emergency Procurement Circular (D020) count as both T2 and T3?** It's a
   genuine temporary *contradiction* with the standing policy (T2), and it also has two boundary
   dates where the answer flips (T3) — real corpora often have overlapping trap types, and we
   didn't force artificial separation just to keep the taxonomy tidy. The same overlap appears
   with the Regional Festive Gifting Circular (D043).
4. **Why do two of the six `conflict`-labelled questions deliberately resist auto-resolution?**
   True same-specificity, cleanly-resolvable contradictions are the easy half of conflict
   detection. In a real organisation, the harder and more common failure mode is a narrower
   circular from a team that doesn't own the policy it's touching — that's a governance problem,
   not a retrieval problem, and the correct system behaviour is to escalate rather than silently
   let "more specific" win. Including both kinds of case (Section 2) keeps the `conflict` label
   from just testing whether specificity was implemented — it also tests whether the system
   knows the limits of that rule.

## 5. What's still open for M2 onward

- The dev set (`gold_qa_dev.jsonl`) is what `eval.py` (Week 3–4) will score the baseline
  pipeline against. Expect the M2 baseline to fail most T1/T3 questions — per the problem
  statement, "that failure is the point" and motivates the temporal filter (M3) and conflict/
  refusal gate (M4).
- `gold_qa_heldout.jsonl` stays sealed until Week 8 (M5).

## Checkpoint status

☐ Dataset review with instructor — confirm the trap taxonomy is genuinely present and
correctly labelled (bring `traps.md` and `gold_qa.jsonl` to this review).
