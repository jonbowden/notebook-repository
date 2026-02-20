"""
Mock Toolkit for Module 8: Agents & Automation

Provides a keyword-based mock LLM client, three agent tools (classify,
retrieve, refuse), and a decision function — everything needed to build
and run the triage agent without external dependencies.

In production, replace MockLLMClient with your Module 6 LLM client
and retrieve_and_answer with your Module 7 RAG pipeline. The agent
code does not change — only the toolkit swaps out.

Review this file to understand how each component works.
"""

import json


# ── Mock LLM Client ─────────────────────────────────────────

class MockLLMClient:
    """Keyword-based mock that returns JSON action strings.

    Same interface as a real LLM client: call .chat(prompt) and
    get a string back.  Deterministic so grading always produces
    the same result.
    """

    def chat(self, prompt):
        # Extract just the user input from the structured prompt.
        # The prompt template is fixed; only the user input varies.
        if "User input: " in prompt:
            user_part = prompt.split("User input: ")[-1].lower()
        else:
            user_part = prompt.lower()

        if any(kw in user_part for kw in ["rate", "inflation", "monetary"]):
            return '{"action": "retrieve_and_answer"}'
        if any(kw in user_part for kw in ["classify", "label", "topic"]):
            return '{"action": "classify_only"}'
        return '{"action": "refuse"}'


# ── Allowed Actions ─────────────────────────────────────────

ALLOWED_ACTIONS = {
    "retrieve_and_answer",   # look up information and produce a grounded answer
    "classify_only",         # label the topic without generating an answer
    "refuse",                # decline to act on the request
}

REFUSAL_TEXT = "Refused: I am not authorised to act on this request."


# ── Tool 1: Classify ────────────────────────────────────────

def classify_only(text):
    """Classify the topic of the input text.

    Returns a topic label if the input matches a known finance category.
    Returns "Topic: Out of Domain" for unrecognised inputs.
    No side effects.
    """
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["rate", "inflation", "monetary", "central bank"]):
        return "Topic: Interest Rates & Monetary Policy"
    if any(kw in text_lower for kw in ["loan", "mortgage", "lending", "credit"]):
        return "Topic: Lending & Credit"
    if any(kw in text_lower for kw in ["stock", "equity", "market", "earning"]):
        return "Topic: Equity Markets"
    return "Topic: Out of Domain"


# ── Tool 2: Retrieve and Answer ─────────────────────────────

KNOWLEDGE_BASE = {
    "rate":      "The central bank raised interest rates by 25 basis points to combat persistent inflation.",
    "inflation": "Inflation remained elevated at 4.2% year-on-year, driven by energy prices.",
    "mortgage":  "Mortgage rates reached a two-decade high, causing home sales to decline 15%.",
    "loan":      "Lending standards tightened as banks responded to higher funding costs.",
    "monetary":  "The dual mandate requires balancing maximum employment with price stability.",
}


def retrieve_and_answer(question):
    """Return a grounded answer from the knowledge base.

    Simulates a Module 7 RAG retrieval without requiring an
    embedding model or vector database.
    """
    question_lower = question.lower()
    for keyword, answer in KNOWLEDGE_BASE.items():
        if keyword in question_lower:
            return f"Based on retrieved documents: {answer}"
    return "Based on retrieved documents: No relevant information found."


# ── Tool 3: Refuse ──────────────────────────────────────────

def refuse():
    """Return the standard refusal message.

    Refusal is a deliberate, safe, terminal action — not a failure.
    """
    return REFUSAL_TEXT


# ── Decision Function ───────────────────────────────────────

def decide_action(llm_client, user_input):
    """Ask the LLM to choose an action, parse JSON, and validate.

    Falls back to 'refuse' on any error:
      - Malformed JSON from the LLM
      - Missing 'action' key
      - Action not in ALLOWED_ACTIONS
    """
    prompt = (
        "Choose exactly one action: retrieve_and_answer, classify_only, refuse\n"
        'Return ONLY valid JSON: {"action": "<action>"}\n'
        f"User input: {user_input}"
    )
    raw = llm_client.chat(prompt)
    try:
        action = json.loads(raw).get("action", "refuse")
    except (json.JSONDecodeError, AttributeError):
        action = "refuse"
    if action not in ALLOWED_ACTIONS:
        action = "refuse"
    return action
