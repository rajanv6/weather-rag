"""Input guardrails, applied before a question costs any LLM or embedding work.

Be clear about what this is: a cheap first screen that stops the obvious abuse
(oversized input, control characters, well-known instruction-override and
secret-fishing phrasings). Keyword filters can always be rephrased around, so
the real controls are structural: secrets are never placed in the model's
context, the tools are read-only and validated, and every upstream call is
rate-limited and budgeted. The system prompt adds a scope rule on top."""

import re
import unicodedata

MAX_QUESTION_CHARS = 500
MAX_BODY_BYTES = 8 * 1024  # a 500-char question is ~2 KB of JSON even if every char is escaped

_INVISIBLE = dict.fromkeys(  # zero-width and bidi characters used to hide words from filters
    [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069]
)

_OVERRIDE = r"(?:previous|prior|above|earlier|preceding|your|all|any|the|these|those|system)"
_INJECTION_PATTERNS = [
    re.compile(p)
    for p in (
        rf"\b(?:ignore|disregard|forget|override|bypass)\b(?:\s+{_OVERRIDE}){{1,4}}\s+(?:instructions?|rules?|prompts?|guidelines?|directions?|constraints?)",
        r"\b(?:reveal|show|print|repeat|display|leak|output|tell\s+me|give\s+me|what(?:'s|\s+is|\s+are))\b.{0,30}\b(?:system|hidden|initial|original|developer)\s+(?:prompt|instructions?|message)",
        r"\b(?:reveal|show|print|leak|dump|display|give\s+me|what(?:'s|\s+is))\b.{0,40}(?:\bapi[\s_-]?keys?\b|\bsecrets?\b|\bpasswords?\b|\bcredentials?\b|\.env\b|\benvironment\s+variables?\b)",
        r"\b(?:jailbreak|developer\s+mode|dan\s+mode|do\s+anything\s+now)\b",
    )
]


def normalize(text: str) -> str:
    """NFKC folds look-alike characters (fullwidth letters etc.); then drop
    invisible characters, lowercase and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE).lower()
    return re.sub(r"\s+", " ", text).strip()


def clean_question(text: str) -> str:
    """Drops control characters (keeps ordinary whitespace) and trims."""
    kept = "".join(ch for ch in text if ch in "\n\t " or unicodedata.category(ch)[0] != "C")
    return kept.strip()


def screen(question: str) -> tuple[str, str | None]:
    """Returns (cleaned_question, problem). `problem` is a user-facing reason
    to reject the request, or None if it may proceed."""
    cleaned = clean_question(question)
    if not cleaned:
        return cleaned, "question is required"
    if len(cleaned) > MAX_QUESTION_CHARS:
        return cleaned, f"question is too long (max {MAX_QUESTION_CHARS} characters)"
    folded = normalize(cleaned)
    if any(p.search(folded) for p in _INJECTION_PATTERNS):
        return cleaned, (
            "that looks like an attempt to override the assistant's instructions "
            "or extract secrets, so it was not sent"
        )
    return cleaned, None
