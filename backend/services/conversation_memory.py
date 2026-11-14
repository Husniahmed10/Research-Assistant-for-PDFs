"""Narrow, deterministic name recall from user-authored session messages."""

import re
import unicodedata

from services.input_policy import blocks_input

_RECALL = re.compile(
    r"(?:what(?:'s|s| is) my name|tell me my name|"
    r"(?:can|could) you tell me my name|(?:do|can) you remember my name|"
    r"remind me (?:of )?my name)[?!.]*", re.IGNORECASE,
)
_GREETING = r"(?:hi|hello|hey|hry)(?:\s+there)?"
_INTRO = re.compile(
    rf"(?:^|[.!?\n])\s*(?:(?:{_GREETING}|actually|sorry|please)(?:[,!]\s*|\s+))?"
    r"(?:my name is|i am|i'm|call me)\s+([^\n.,;:!?]+)", re.IGNORECASE,
)
_NOT_NAMES = {
    "a", "an", "the", "not", "so", "very", "here", "using", "trying", "studying",
    "working", "looking", "interested", "asking", "happy", "sorry", "ignore", "unsure",
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u2019", "'").strip()


def is_name_recall_query(query: str) -> bool:
    text = " ".join(_normalize(query).split())
    text = re.sub(rf"^{_GREETING}\b[,!.\s]*", "", text, count=1, flags=re.IGNORECASE)
    return _RECALL.fullmatch(text) is not None


def conversation_reply(query: str, name: str | None = None) -> str | None:
    """Handle harmless conversation without turning it into a retrieval task."""
    if is_name_recall_query(query):
        return f"Your name is {name}." if name else "I couldn't find your name in this chat's saved messages."
    text = " ".join(_normalize(query).split()).lower().strip(" .!?,")
    if re.fullmatch(_GREETING, text):
        return "Hi! What would you like to know about the uploaded paper?"
    if text in {"thanks", "thank you", "thanks a lot", "thank you very much"}:
        return "You're welcome! Ask me if you have more questions about the paper."
    if text in {"ok", "okay", "got it"}:
        return "Okay. What else would you like to explore in the paper?"
    if text in {"bye", "goodbye"}:
        return "Goodbye! You can return to this chat for more questions about the paper."
    introduction = _INTRO.fullmatch(_normalize(query).rstrip(".!?"))
    if introduction:
        introduced_name = extract_user_name([query])
        # Do not swallow a paper question attached to an introduction.
        if introduced_name and " ".join(introduction.group(1).split()) == introduced_name:
            return f"Nice to meet you, {introduced_name}. What would you like to know about the paper?"
    return None


def extract_user_name(user_messages: list[str]) -> str | None:
    """Use the most recent simple introduction; never infer from assistant text.

    This is a display name supplied by the user, not identity verification or
    a general profile store. Restrict output to short alphabetic name phrases.
    """
    for content in reversed(user_messages):
        content = _normalize(content)
        if blocks_input(content):
            continue
        for match in reversed(list(_INTRO.finditer(content))):
            candidate = re.split(r"\s+(?:and|who|what|can|please)\b", match.group(1), maxsplit=1,
                                 flags=re.IGNORECASE)[0]
            words = candidate.split()
            if not 1 <= len(words) <= 6 or len(candidate) > 100:
                continue
            if words[0].lower() in _NOT_NAMES:
                continue
            if all(word[0].isalpha() and word[-1].isalpha()
                   and all(char.isalpha() or char in "-'" for char in word) for word in words):
                return " ".join(words)
    return None
