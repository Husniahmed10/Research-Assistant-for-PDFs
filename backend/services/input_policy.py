"""High-confidence instruction overrides, checked locally before any model call.

Patterns target commands at the start of a sentence, rather than mentions in
paper questions. Semantic/off-topic checks remain the responsibility of NeMo.
"""

import re
import unicodedata


_START = r"(?:^|[.!?\n]\s*)(?:(?:hi|hello|hey)[,!]?\s+)?(?:please\s+)?"
_JAILBREAK = re.compile(_START + r"(?:"
    r"(?:ignore|disregard|forget|bypass|override)\s+(?:all\s+)?(?:the\s+|your\s+)?"
    r"(?:(?:previous|prior|system|safety|above|earlier)\s+)*(?:instructions|rules|guidelines|restrictions|filters)\b"
    r"|(?:you are now|act as|pretend to be)\s+(?:an?\s+)?(?:DAN|unrestricted|uncensored|unfiltered)\b"
    r"|(?:enable|activate|enter)\s+(?:developer|DAN|jailbreak)\s+mode\b"
    r"|(?:disable|remove|turn off)\s+(?:all\s+|your\s+)?(?:safety|security)\s+(?:checks|filters|rules)\b"
    r")", re.IGNORECASE)
_INJECTION = re.compile(_START + r"(?:"
    r"(?:reveal|print|show|repeat|expose|give me)\s+(?:the\s+|your\s+)?"
    r"(?:hidden\s+|secret\s+)?(?:system prompt|developer instructions|API keys?|passwords?|credentials)\b"
    r"|(?:new|replacement)\s+system\s+(?:prompt|instructions)\s*:"
    r"|system\s*:|<\|im_start\|>\s*system|\[INST\]|<<SYS>>"
    r"|(?:treat|follow)\s+(?:this|the following)\s+as\s+(?:a\s+|the\s+)?system\s+(?:message|prompt)\b"
    r")", re.IGNORECASE)


def _normalize(query: str) -> str:
    return unicodedata.normalize("NFKC", query).strip()


def is_jailbreak(query: str) -> bool:
    return bool(_JAILBREAK.search(_normalize(query)))


def is_prompt_injection(query: str) -> bool:
    return bool(_INJECTION.search(_normalize(query)))


def blocks_input(query: str) -> bool:
    return is_jailbreak(query) or is_prompt_injection(query)
