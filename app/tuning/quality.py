"""Fast output quality checks."""

import re

MIN_WORDS = 3
MIN_SENTENCE_CHARS = 3

GARBAGE_PATTERNS = [
    r"<\|[^|]+\|>",           # <|text|>, <|im_start|>, etc.
    r"[�]",               # replacement char
]

def detect_garbage(text: str) -> bool:
    if not text or len(text.strip()) == 0:
        return True
    for p in GARBAGE_PATTERNS:
        if re.search(p, text):
            return True
    special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    return special_ratio > 0.3

def is_valid_response(text: str) -> tuple[bool, str]:
    if not text or len(text.strip()) < MIN_WORDS * 4:
        return False, "too_short"
    if detect_garbage(text):
        return False, "special_token_leak"
    sentences = [s for s in re.split(r'[.!?]', text) if len(s.strip()) > MIN_SENTENCE_CHARS]
    if len(sentences) < 2:
        return False, "not_coherent"
    words = text.lower().split()
    for i in range(len(words) - 8):
        phrase = " ".join(words[i:i+4])
        if phrase in " ".join(words[i+4:min(i+16, len(words))]):
            return False, "phrase_repetition"
    return True, "ok"