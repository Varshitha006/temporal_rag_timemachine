# traps.md — Planted Trap Ledger
Corpus: Solstice Robotics Pvt Ltd (fictional company), 44 documents, 2019–2024/25.

This ledger documents every trap deliberately planted in the corpus, per the taxonomy in
the problem statement (A.4). Each row says **where** the trap lives, **what** it tests, and
**which gold questions** exercise it, so the evaluation harness (M2) can report accuracy
per trap type, not just an aggregate score.

---

## T1 — Superseded-but-similar
An old clause is the closest semantic/keyword match to a common query, but is no longer in force.

| # | Where planted | Why it's a trap | Gold Qs |
|---|---|---|---|
| T1-a | WFH Policy chain D001→D002→D003 | All three use near-identical phrasing ("work from home", "days per week"), but the rules changed materially (2 days → 4 days → office-mandatory hybrid). A generic "what is the WFH policy" query with no date will match D001 most closely on raw term overlap even though it expired in 2022. | Q004 |
| T1-b | Password Policy chain D012→D013→D014 | "Password", "change every N days" appears in D012/D013 but the *current* policy (D014) removed periodic changes entirely and added a completely different concept (MFA-mandatory). Retrieval keyed on "password change" will surface expired text. | Q005 |
| T1-c | Leave Policy D006 "sick leave: 12 days" vs D007 (merged into Flexible Leave) | The literal term "sick leave" only appears in the retired D005/D006; the current D007 doesn't use the term at all, so naive retrieval either returns a stale number or (worse) fails silently. | Q022 |
| T1-d | Procurement quotation threshold D018 vs D019 | The number "2,00,000" is is the threshold in the *old* policy for quotations; the current policy raised it to 3,00,000. A query near the boundary must not anchor on the more "familiar"/older number. | Q037 |
| T1-e | Equipment Allowance D030 (₹8,000) vs D031 (₹6,000) | ₹8,000 is the larger, more "salient" number and was in force for 18 months (longer than v1); a system might over-weight it as the answer even after v3.0 superseded it. | Q058 |

## T2 — Contradiction (two in-force documents conflict)
Two documents are simultaneously valid but disagree on the same point; the system must detect
the conflict and resolve it via precedence (specificity / legal mandate / explicit override), not
silently pick one. **Not every conflict should auto-resolve** — T2-d and T2-f below are
deliberately *not* cleanly resolvable by the specificity rule alone, and are included to test
whether the system over-applies precedence logic in cases where it shouldn't.

| # | Where planted | Precedence rule that resolves it | Gold Qs |
|---|---|---|---|
| T2-a | Data Retention Policy v2.0 (D010, general: 3 years) vs GDPR Addendum (D011, EU personal data: max 1 year, explicitly says it overrides) | **Specificity + legal mandate**: the addendum is narrower in scope (EU data only) and cites a legal basis, so it wins for EU data; the general policy still governs everything else. | Q009, Q010 |
| T2-b | Procurement Policy v2.0 (D019, Finance Head ceiling ₹5,00,000) vs Emergency Procurement Circular (D020, temporarily raises ceiling to ₹10,00,000, 2023-08-01 to 2023-10-31) | **Explicit temporary override + recency-bounded validity**: the circular is a time-boxed amendment that supersedes the standing policy only within its window, then automatically reverts. | Q011, Q012 |
| T2-c | International Travel Addendum (D017, general: USD 60/day) vs Client-Site International Travel Circular (D041, USD 45/day for on-site client project work) | **Specificity, cleanly resolved**: D041 is narrower (client-site work specifically) and wins for that subset; D017 continues to govern all other international travel. | Q061 |
| T2-d | Data Classification Policy (D034, NDA required for Confidential data shared externally) vs Sales Enablement Confidentiality Circular (D042, Sales & Marketing, no NDA needed to share roadmaps with prospects) | **Deliberately NOT auto-resolved by specificity.** D042 is narrower in scope, but it was issued by a non-compliance-owning business team and directly weakens a compliance-critical control. Tests whether the system correctly refuses to let a "more specific" business circular silently override a security/compliance policy it doesn't own — correct behaviour is to flag and escalate to Legal & Compliance / IT Governance. | Q062 |
| T2-e | Gifts & Anti-Bribery Policy v2.0 (D036, ₹3,000 pre-approval threshold) vs Regional Festive Gifting Circular (D043, ₹7,000 threshold, South Region only, 2024-10-01 to 2024-11-15) | **Explicit time-boxed exception + specificity, cleanly resolved** — same pattern as T2-b. | Q063 |
| T2-f | Password & MFA Policy v3.0 (D014, MFA mandatory "with no exceptions") vs Field Operations Device Exemption Circular (D044, exempts shared kiosk devices) | **Deliberately NOT auto-resolved by specificity.** D044 was issued by Field Operations, not IT Security (owner of the MFA mandate), and tries to carve an exception into an explicit "no exceptions" clause. Correct behaviour is escalation to IT Security, not silent acceptance of the exemption. | Q064 |

## T3 — Temporal edge (answer flips as as_of_date crosses a boundary)
| # | Where planted | Boundary date(s) | Gold Qs |
|---|---|---|---|
| T3-a | MFA mandatory only from D014's effective date | 2024-07-01 | Q006, Q007, Q008 (exact boundary) |
| T3-b | Emergency Procurement Circular window | 2023-08-01 / 2023-10-31 | Q011, Q012 |
| T3-c | Special COVID Leave Circular window | 2021-04-01 / 2021-12-31 | Q023, Q024 |
| T3-d | BYOD lost-device reporting rule did not exist before v2.0 | 2024-01-01 | Q043 |
| T3-e | Equipment allowance amount changes across three windows | 2022-07-01 / 2024-02-01 | Q058, Q059 |

## T4 — Silent / missing (corpus genuinely does not answer)
| # | Question area | Why it's genuinely unanswerable | Gold Qs |
|---|---|---|---|
| T4-a | Sabbatical leave | No document of any kind addresses sabbaticals. | Q013 |
| T4-b | Cryptocurrency purchases | Not covered by Procurement or Finance policies. | Q014 |
| T4-c | Gym membership reimbursement | Not covered by T&E or any HR circular. | Q015 |
| T4-d | Adoption leave *before* 2023-07-01 | The provision did not exist yet in the policy in force on that date (D037 is silent); this is a "silent-at-that-date" variant of T4, distinct from "never covered anywhere." | Q051 |
| T4-e | Facilitation payments *before* 2024-01-01 | Same pattern — the prohibition was only added in D036 (v2.0); D035 (v1.0, in force earlier) does not mention it. | Q056 |
| T4-f | Co-working space stipends | Not covered by WFH policy or Equipment Allowance circulars, which only address home-based costs. | Q065 |
| T4-g | Pets in the office | Not covered by any document. | Q066 |
| T4-h | Employee referral bonus program | Not covered by any document, including Background Verification and Exit policies. | Q067 |

Note: D040 (Company Values & Mission Statement) is a deliberate **decoy document** — it is
topically adjacent to almost every HR question but explicitly states it is non-binding and must
not be cited as governing authority. Any T4 question should never resolve to D040 as its citation.

## T5 — Ambiguous scope (a rule for one population queried for another)
| # | Where planted | Ambiguity | Gold Qs |
|---|---|---|---|
| T5-a | WFH Policy (employees only) vs Contractor Engagement Policy ("unless stated in contract") | Whether a *specific* contractor can WFH depends on their individual contract, which is outside the corpus — correct behaviour is to ask/escalate, not assert yes/no. | Q016 |
| T5-b | Regional WFH Exception Circular (D004, silent on employee type) vs general contractor exclusion | The circular doesn't say "employees only," but it's framed as an exception to a policy that does exclude contractors. | Q017 |
| T5-c | Equipment Allowance v3.0 eligibility vs undefined intern status | Neither the WFH policy nor the allowance circular defines intern eligibility explicitly. | Q018 |
| T5-d | Leave Policy v3.0 (D007) states an entitlement but never specifies employee category (full-time vs part-time) | Unlike the WFH policy, which explicitly scopes itself to full-time employees, the Leave Policy is silent on this — silence is not itself an answer. | Q068 |
| T5-e | Whistleblower Policy (D033) uses "employee" throughout without defining intern status | Same ambiguous-population pattern applied to a protection clause rather than a benefit. | Q069 |
| T5-f | BYOD Policy (D024) doesn't address contractors; Contractor Engagement Policy (D025) is silent on device rules | Two documents, each silent on the exact question from a different angle — neither confirms nor excludes. | Q070 |
| T5-g | Parental Leave Policy v2.0 (D038) has no probation-completion condition, unlike the WFH policy | Absence of an exclusion clause is not proof of inclusion. | Q071 |
| T5-h | Regional Festive Gifting Circular (D043) scoped to "South Region sales employees" — undefined for a cross-regional role | Population boundary ambiguity at the edge of an org-chart definition, not a temporal or precedence question. | Q072 |

---

## Coverage summary

| Trap type | Count in gold set | Docs involved |
|---|---|---|
| T1 | 5 | D001-D003, D005-D007, D012-D014, D018-D019, D030-D031 |
| T2 | 6 | D010-D011, D019-D020, D017/D041, D034/D042, D036/D043, D014/D044 |
| T3 | 8 | D008, D013-D014, D019-D020, D023-D024, D029-D031 |
| T4 | 8 | (absence is the point) + D037/D035→D038/D036 silent-at-date variants + pure-absence questions (co-working stipend, pets, referral bonus) |
| T5 | 8 | D003-D004, D025, D007, D033, D024/D025, D038, D043 |
| none (control) | 45 | spread across all 44 corpus documents |

**Total gold questions: 80** (42 → 62 answerable-family, refuse 9, escalate 8, conflict 6).
**Split: 56 dev / 24 held-out (exact 70/30).**

All five trap types (T1-T5) required by the problem statement are present, each with
≥ 5 gold questions, split across both the dev and held-out sets (see `gold_qa.jsonl`,
field `split`). T2 and T5 in particular now include both "cleanly resolvable by precedence"
and "should NOT be auto-resolved, must escalate" instances — this distinction is deliberate
and is the harder half of the M4 decision-gate logic to get right.
