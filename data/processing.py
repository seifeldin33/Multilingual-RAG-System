"""
Dataset loading, cleaning, text chunking, and metadata enrichment.

Processes the Natural Questions dataset into structured Document objects
ready for embedding and indexing.
"""

import re
import logging
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional

import pandas as pd
from tqdm import tqdm

from config.settings import settings

logger = logging.getLogger(__name__)


# ── HTML Stripping ────────────────────────────────────────────────────────────

class _HTMLTextExtractor(HTMLParser):
    """Lightweight HTML-to-text converter (no external dependency on bs4)."""

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._pieces.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw = "".join(self._pieces)
        # Collapse multiple whitespace / blank lines
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def strip_html(text: str) -> str:
    """Remove HTML tags and return clean text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(text)
        return extractor.get_text()
    except Exception:
        # Fallback: regex-based stripping
        return re.sub(r"<[^>]+>", " ", text).strip()


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single processed document chunk with metadata."""

    id: int
    question: str
    answer_text: str          # full cleaned answer
    chunk_text: str           # the text chunk used for embedding
    chunk_index: int          # position of this chunk within the answer
    total_chunks: int         # total chunks for this answer
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Document(id={self.id}, q='{self.question[:40]}…', "
            f"chunk={self.chunk_index}/{self.total_chunks})"
        )


# ── Question-Type Classification ─────────────────────────────────────────────

_QUESTION_PATTERNS: list[tuple[str, str]] = [
    (r"^(who|whom)\b",   "person"),
    (r"^what\b",         "factoid"),
    (r"^when\b",         "temporal"),
    (r"^where\b",        "location"),
    (r"^why\b",          "explanation"),
    (r"^how many\b",     "numeric"),
    (r"^how much\b",     "numeric"),
    (r"^how\b",          "process"),
    (r"^which\b",        "selection"),
    (r"^(is|are|was|were|do|does|did|can|could|will|would|should)\b", "yes_no"),
]


def classify_question_type(question: str) -> str:
    """Classify a question into a semantic type using regex patterns."""
    q = question.strip().lower()
    for pattern, qtype in _QUESTION_PATTERNS:
        if re.search(pattern, q):
            return qtype
    return "other"


# ── Domain Classification ────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "science":        ["atom", "molecule", "cell", "dna", "gene", "planet", "element",
                       "energy", "physics", "chemistry", "biology", "evolution", "quantum"],
    "history":        ["war", "empire", "century", "king", "queen", "dynasty", "battle",
                       "revolution", "ancient", "medieval", "historical", "colony"],
    "geography":      ["country", "continent", "ocean", "river", "mountain", "capital",
                       "population", "border", "island", "desert", "climate"],
    "entertainment":  ["movie", "film", "song", "album", "actor", "actress", "singer",
                       "series", "show", "game", "award", "oscar", "grammy"],
    "sports":         ["team", "player", "championship", "goal", "score", "league",
                       "tournament", "olympic", "match", "coach", "season", "cup"],
    "technology":     ["computer", "software", "internet", "app", "programming", "data",
                       "algorithm", "digital", "network", "server", "cloud"],
    "politics":       ["president", "minister", "government", "election", "law", "policy",
                       "congress", "parliament", "vote", "political", "democrat", "republican"],
    "health":         ["disease", "symptom", "treatment", "doctor", "hospital", "medicine",
                       "virus", "bacteria", "vaccine", "surgery", "patient", "health"],
}


def classify_domain(text: str) -> str:
    """Tag a document with a domain based on keyword overlap."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


# ── Difficulty Classification ─────────────────────────────────────────────────

def classify_difficulty(answer_text: str, question: str) -> str:
    """Estimate difficulty from answer length and question complexity."""
    answer_len = len(answer_text)
    question_words = len(question.split())
    if answer_len < 100 and question_words < 10:
        return "easy"
    elif answer_len < 500:
        return "medium"
    else:
        return "hard"


# ── Text Chunking ─────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[str]:
    """
    Split *text* into overlapping chunks of at most *chunk_size* characters.

    Tries to break on sentence boundaries (`. `) first, falling back to
    word boundaries, then hard character splits.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Prefer sentence boundary
            boundary = text.rfind(". ", start, end)
            if boundary == -1 or boundary <= start:
                # Fall back to word boundary
                boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary + 1  # include the space / period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance with overlap
        start = max(start + 1, end - overlap)

    return chunks


# ── Metadata Builder ──────────────────────────────────────────────────────────

def _build_metadata(
    question: str,
    answer_text: str,
    has_short: bool,
    has_long: bool,
) -> dict:
    combined = f"{question} {answer_text}"
    return {
        "question_type": classify_question_type(question),
        "domain": classify_domain(combined),
        "difficulty": classify_difficulty(answer_text, question),
        "has_short_answer": has_short,
        "has_long_answer": has_long,
        "answer_length": len(answer_text),
        "question_length": len(question),
    }


# ── Main Processing Pipeline ─────────────────────────────────────────────────

def load_and_clean(
    path: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the Natural Questions CSV, drop fully-null rows, and clean HTML.

    Parameters
    ----------
    path : str, optional
        Override for the dataset path (defaults to settings).
    sample_size : int, optional
        Number of rows to sample (0 or None = full dataset).

    Returns
    -------
    pd.DataFrame  with columns: question, long_answer, short_answer
    """
    path = path or str(settings.dataset_full_path)
    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path)
    logger.info("Raw dataset: %d rows, columns: %s", len(df), list(df.columns))

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename to consistent names if needed
    col_map = {}
    for col in df.columns:
        if "long" in col and "answer" in col:
            col_map[col] = "long_answer"
        elif "short" in col and "answer" in col:
            col_map[col] = "short_answer"
    if col_map:
        df.rename(columns=col_map, inplace=True)

    # Drop rows where BOTH answers are null
    before = len(df)
    df = df.dropna(subset=["long_answer", "short_answer"], how="all").reset_index(drop=True)
    logger.info("Dropped %d rows with no answers → %d rows remaining", before - len(df), len(df))

    # Sample if requested
    sample_size = sample_size if sample_size is not None else settings.sample_size
    if sample_size and 0 < sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info("Sampled %d rows for processing", sample_size)

    # Clean HTML in long answers
    logger.info("Stripping HTML from long answers …")
    df["long_answer"] = df["long_answer"].apply(strip_html)

    # Fill remaining NaN with empty strings
    df["long_answer"] = df["long_answer"].fillna("")
    df["short_answer"] = df["short_answer"].fillna("")

    return df


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert a cleaned DataFrame into a list of Document chunks.

    Each row may produce multiple Document objects if its answer text is
    longer than the configured chunk size.
    """
    documents: list[Document] = []
    doc_id = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building documents"):
        question = str(row["question"]).strip()
        long_answer = str(row.get("long_answer", "")).strip()
        short_answer = str(row.get("short_answer", "")).strip()

        # Use long answer if available, else short answer
        answer_text = long_answer if long_answer else short_answer
        if not answer_text:
            continue

        has_short = bool(short_answer)
        has_long = bool(long_answer)

        # Compute metadata once per row
        metadata = _build_metadata(question, answer_text, has_short, has_long)

        # Chunk the answer
        chunks = chunk_text(answer_text)
        if not chunks:
            continue

        for i, chunk in enumerate(chunks):
            # Prepend question context to the chunk for better embedding quality
            contextualised = f"Question: {question}\nAnswer: {chunk}"
            documents.append(
                Document(
                    id=doc_id,
                    question=question,
                    answer_text=answer_text,
                    chunk_text=contextualised,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    metadata=metadata,
                )
            )
            doc_id += 1

    logger.info(
        "Built %d document chunks from %d rows (avg %.1f chunks/row)",
        len(documents),
        len(df),
        len(documents) / max(len(df), 1),
    )
    return documents


def process_dataset(
    path: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> list[Document]:
    """End-to-end: load → clean → chunk → enrich metadata → return Documents."""
    df = load_and_clean(path=path, sample_size=sample_size)
    return build_documents(df)
