# app/tools/snippets.py
"""
Snippet extraction and BM25-inspired relevance scoring for evidence retrieval.
"""
import re

# ---------------------------------------------------------------------------
# Stopwords (compact English set for fact-checking context)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "not", "no", "nor",
    "so", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "he", "she", "they", "them", "his", "her", "we", "us", "you",
    "your", "my", "i", "me", "as", "up", "out", "about", "into", "over",
    "after", "before", "between", "under", "again", "there", "here",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "too",
    "very", "just", "also", "now", "s", "t", "ve", "re", "ll", "d", "m",
    "said", "says", "like", "get", "got", "go", "went", "come", "came",
    "make", "made", "know", "known", "see", "seen", "think", "thought",
    "well", "back", "even", "still", "way", "because", "through", "much",
    "while", "what", "which", "who", "whom", "whose", "one", "two",
    "according", "speaker", "claim", "claims", "states", "statement",
})

_TOKEN_RE = re.compile(r"[A-Za-z0-9%$]+")
_NUMERIC_RE = re.compile(r"^\d[\d,.%$]*$")


def _tokenize(text: str) -> list:
    """Tokenize and lowercase, keeping numbers, % and $ signs."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _tokenize_no_stop(text: str) -> list:
    """Tokenize, lowercase, and remove stopwords."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Snippet chunking
# ---------------------------------------------------------------------------

def make_snippets(text: str, max_chars: int = 1000, overlap: int = 200):
    """Split text into overlapping chunks. Returns [(start, end, chunk_text), ...]."""
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


# ---------------------------------------------------------------------------
# BM25-inspired scoring
# ---------------------------------------------------------------------------

# BM25 parameters tuned for short snippets (500-1200 chars)
_K1 = 1.2
_B = 0.75
_AVG_DOC_LEN = 150.0  # approximate average snippet length in tokens


def score_snippet(claim_text: str, snippet_text: str) -> float:
    """
    BM25-inspired relevance scoring.

    Improvements over naive token overlap:
    - Stopword removal: "the", "is", "a" no longer inflate scores
    - TF saturation: a term appearing 10x is not scored 10x higher than 1x
    - Numeric boost: numbers/percentages get 3x IDF weight (critical for fact-checking)
    - Length normalization: shorter snippets with matching terms score higher

    Returns 0.0-1.0 normalized score.
    """
    claim_tokens = _tokenize_no_stop(claim_text)
    if not claim_tokens:
        return 0.0

    snippet_tokens = _tokenize(snippet_text)
    if not snippet_tokens:
        return 0.0

    # Build snippet term frequencies
    snippet_tf = {}
    for t in snippet_tokens:
        snippet_tf[t] = snippet_tf.get(t, 0) + 1

    snippet_len = len(snippet_tokens)

    # Unique claim terms
    claim_term_set = set(claim_tokens)

    def _idf(term):
        """Approximate IDF. Numbers and percentages get boosted."""
        if _NUMERIC_RE.match(term):
            return 3.0
        return 1.0

    score = 0.0
    for term in claim_term_set:
        tf = snippet_tf.get(term, 0)
        if tf == 0:
            continue
        idf = _idf(term)
        # BM25 TF saturation with length normalization
        tf_norm = (tf * (_K1 + 1)) / (tf + _K1 * (1 - _B + _B * (snippet_len / _AVG_DOC_LEN)))
        score += idf * tf_norm

    # Normalize by max possible score (all terms present, tf=1, avg length)
    max_possible = sum(_idf(t) for t in claim_term_set) * (_K1 + 1) / (1 + _K1 * (1 - _B + _B * (snippet_len / _AVG_DOC_LEN)))
    if max_possible > 0:
        score = score / max_possible

    return min(1.0, score)


def top_k_snippets(claim_text: str, snippets, k: int = 5):
    """Return top-k snippets scored by BM25-inspired relevance."""
    scored = []
    for (start, end, chunk) in snippets:
        s = score_snippet(claim_text, chunk)
        scored.append((s, start, end, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]
