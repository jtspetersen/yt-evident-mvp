# app/tools/snippets.py
import re

def make_snippets(text: str, max_chars: int = 1000, overlap: int = 200):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunk = text[i:j]
        chunks.append((i, j, chunk))
        i = max(0, j - overlap)
        if j == n:
            break
    return chunks

def score_snippet(claim_text: str, snippet_text: str) -> float:
    tokens = set(re.findall(r"[A-Za-z0-9%]+", claim_text.lower()))
    if not tokens:
        return 0.0
    stokens = set(re.findall(r"[A-Za-z0-9%]+", snippet_text.lower()))
    overlap = tokens.intersection(stokens)
    return len(overlap) / max(1, len(tokens))

def top_k_snippets(claim_text: str, snippets, k: int = 5):
    scored = []
    for (start, end, chunk) in snippets:
        s = score_snippet(claim_text, chunk)
        scored.append((s, start, end, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]
