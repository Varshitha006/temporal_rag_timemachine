"""
llm_client.py
--------------
Google Gemini API client for final answer generation.

Uses the new `google.genai` SDK (replaces deprecated `google.generativeai`).

Reads credentials from .env:
  LLM_API_KEY   -> Google API key
  LLM_MODEL     -> e.g. "gemini-1.5-flash" or "gemini-2.0-flash"

The LLM is called ONLY in the final step -- after retrieval, temporal filtering,
and conflict detection have produced a structured DecisionResult.

The prompt instructs the model to:
  - Answer faithfully using only cited clauses.
  - State the as_of_date used and which document governs.
  - For 'conflict': surface both documents and explain precedence reasoning.
  - For 'escalate' / 'refuse': clearly state why and what information is needed.
  - NEVER hallucinate beyond what the cited clauses state.
"""

import os

from dotenv import load_dotenv

from src.conflict_detector import DecisionResult
from src.temporal_filter import FilteredClause

# Load environment variables from .env
load_dotenv()


# ── Model name normalisation ──────────────────────────────────────────────────
# Map user-friendly model aliases from .env to active, available Gemini API models.
MODEL_ALIASES = {
    # Gemini 3.5 series (fast and responsive, active free tier quota)
    "gemini-3.5-flash":      "gemini-3.5-flash",
    "gemini-3.5-flash-lite": "gemini-3.5-flash-lite",

    # Gemini 3.6 & 3.8 series
    "gemini-3.6-flash":      "gemini-3.6-flash",
    "gemini-3.8-flash":      "gemini-3.8-flash",

    # Generic short aliases mapped to the best active flash model
    "gemini-flash":          "gemini-3.5-flash",
    "gemini-flash-lite":     "gemini-3.5-flash-lite",
    "gemini-pro":            "gemini-3.1-pro-preview",

    # Backward compatibility mappings for older version names
    "gemini-2.5-flash":      "gemini-3.5-flash",
    "gemini-2.5-pro":        "gemini-3.1-pro-preview",
    "gemini-2.0-flash":      "gemini-3.5-flash",
    "gemini-1.5-flash":      "gemini-3.5-flash",
    "gemini-1.5-pro":        "gemini-3.1-pro-preview",
}


def _resolve_model(name: str) -> str:
    """Map alias model names to real Gemini API model names."""
    return MODEL_ALIASES.get(name.lower().strip(), name.strip())


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are TimeMachine, a temporal policy-compliance assistant for Solstice Robotics Pvt Ltd.\n"
    "Your ONLY knowledge source is the policy corpus provided in each query.\n"
    "Rules you MUST follow:\n"
    "1. Cite every claim with a specific clause ID (e.g. D003:C1).\n"
    "2. State the as_of_date you used and which document was in force on that date.\n"
    "3. NEVER guess or infer policy from domains not covered in the corpus.\n"
    "4. For CONFLICT questions: surface all conflicting documents and explain the precedence resolution.\n"
    "5. For ESCALATE questions: clearly state the ambiguity and what clarification is needed.\n"
    "6. For REFUSE questions: explicitly state that the corpus does not address this topic.\n"
)

# ── Answer prompt template ────────────────────────────────────────────────────

ANSWER_TEMPLATE = """\
QUERY DATE: {query_date}
DECISION: {label}
ANSWER HINT: {answer_hint}

GOVERNING CLAUSES (cite these in your answer):
{clauses_block}

DECISION LOG (use to inform your reasoning, do NOT repeat verbatim):
{decision_log}

USER QUESTION: {question}

Write a comprehensive, descriptive compliance answer based strictly on the governing clauses provided above:
1. Direct Answer: State clearly whether the requested action is permitted, prohibited, or subject to specific conditions as of {query_date}.
2. Detailed Policy Explanation: Describe the specific rules, limits, eligibility criteria, and requirements stated in the policies.
3. Citations & Documents: Cite each supporting clause ID (format: [D0xx:Cnn]) alongside the document name, version, and effective date.
4. Edge Cases:
   - For CONFLICT: explain the conflicting provisions and how precedence resolves them.
   - For ESCALATE: specify the missing information needed from the user.
   - For REFUSE: explain that the corpus does not cover the requested topic.
"""


def _build_clauses_block(governing_clauses: list[FilteredClause]) -> str:
    """Format governing clauses into a readable block for the LLM prompt."""
    lines = []
    seen = set()
    for fc in governing_clauses:
        cid = fc.clause.clause_id
        if cid in seen:
            continue
        seen.add(cid)
        meta = fc.clause.meta
        lines.append(
            f"[{cid}] ({meta.doc_id} {meta.version}, "
            f"effective {meta.effective_date} to {meta.sunset_date or 'current'}, "
            f"issued by {meta.issuing_authority})\n"
            f"  Text: {fc.clause.text}\n"
            f"  Status: {fc.precedence_note}"
        )
    return "\n\n".join(lines) if lines else "(no governing clauses -- corpus is silent)"


# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Wraps the Google Gemini API (google.genai SDK) for final answer generation.

    Usage:
        client = LLMClient()
        answer = client.generate(question, query_date_str, decision_result)
    """

    def __init__(self):
        api_key = os.getenv("LLM_API_KEY", "")
        model_env = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        self.model_name = _resolve_model(model_env)

        if not api_key:
            raise ValueError("LLM_API_KEY not set in environment. Check your .env file.")

        # Try the new google.genai SDK first, fall back to old SDK
        try:
            from google import genai
            from google.genai import types as genai_types
            self._client = genai.Client(api_key=api_key)
            self._sdk = "genai"
            print(f"[LLMClient] Using google.genai SDK, model: {self.model_name}")
        except ImportError:
            # Fallback to old SDK if new one not installed yet
            import google.generativeai as genai_old
            genai_old.configure(api_key=api_key)
            self._old_model = genai_old.GenerativeModel(
                model_name=self.model_name,
                system_instruction=SYSTEM_PROMPT,
            )
            self._sdk = "generativeai"
            print(f"[LLMClient] Using google.generativeai SDK (legacy), model: {self.model_name}")

    def generate(
        self,
        question: str,
        query_date_str: str,
        decision: DecisionResult,
    ) -> str:
        """
        Generate a natural-language answer given the structured decision result.

        Args:
            question:       The original user question.
            query_date_str: ISO date string for the query's as_of_date.
            decision:       DecisionResult from the conflict_detector.

        Returns:
            A string answer from the LLM, or a structured fallback if API fails.
        """
        clauses_block   = _build_clauses_block(decision.governing_clauses)
        decision_log_str = "\n".join(decision.decision_log)

        prompt = ANSWER_TEMPLATE.format(
            query_date=query_date_str,
            label=decision.label.upper(),
            answer_hint=decision.answer_hint,
            clauses_block=clauses_block,
            decision_log=decision_log_str,
            question=question,
        )

        # List of models to try in sequence: starts with configured model,
        # then falls back to other fast models if quota or temporary error occurs
        models_to_try = [
            self.model_name,
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.1-flash-lite",
        ]
        # Remove any duplicates while preserving the order of preference
        models_to_try = list(dict.fromkeys(models_to_try))

        last_error = None
        for target_model in models_to_try:
            try:
                if self._sdk == "genai":
                    from google.genai import types as genai_types

                    # Configure generation parameters:
                    # - max_output_tokens=2048: allows full descriptive paragraphs without truncation
                    # - thinking_budget=0: disables internal reasoning tokens so the full token budget is given to the final answer
                    # - temperature=0.1: keeps output factual and strictly bound to the policy text
                    gen_config = genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=2048,
                        temperature=0.1,
                        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                    )

                    # Generate the response using google.genai Client
                    response = self._client.models.generate_content(
                        model=target_model,
                        contents=prompt,
                        config=gen_config,
                    )
                    return response.text.strip()
                else:
                    import google.generativeai as genai_old
                    model_obj = genai_old.GenerativeModel(
                        model_name=target_model,
                        system_instruction=SYSTEM_PROMPT,
                    )
                    response = model_obj.generate_content(prompt)
                    return response.text.strip()
            except Exception as e:
                last_error = e
                continue

        # Graceful structured fallback if all LLM models fail
        return (
            f"[LLM call failed: {last_error}]\n\n"
            f"**Decision:** {decision.label.upper()}\n"
            f"**Hint:** {decision.answer_hint}\n\n"
            f"**Governing clauses:**\n{clauses_block}"
        )
