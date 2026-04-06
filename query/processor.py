"""
Query preprocessing, expansion, and classification.

Provides normalization, synonym expansion via NLTK WordNet,
multi-turn context management, and question-type routing.
"""

import logging
import re
from typing import Optional

import nltk

from config.settings import settings
from data.processing import classify_question_type

logger = logging.getLogger(__name__)

# Ensure WordNet data is available
try:
    from nltk.corpus import wordnet
    wordnet.synsets("test")  # trigger download check
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet


class QueryProcessor:
    """Preprocess and expand user queries for better retrieval."""

    def __init__(self, max_history: int = 5):
        self._history: list[dict[str, str]] = []
        self._max_history = max_history

    # ── Normalisation ─────────────────────────────────────────────────────

    @staticmethod
    def normalize(query: str) -> str:
        """Lowercase, strip excess whitespace, and remove trailing punctuation."""
        q = query.strip().lower()
        q = re.sub(r"\s+", " ", q)
        # Remove trailing question marks / periods (they don't help retrieval)
        q = q.rstrip("?.!")
        return q

    # ── Synonym Expansion ─────────────────────────────────────────────────

    @staticmethod
    def expand_with_synonyms(query: str, max_synonyms_per_word: int = 2) -> str:
        """
        Expand the query by appending WordNet synonyms of key content words.

        Stops words (< 4 chars) are skipped. Duplicates are removed.
        """
        words = query.split()
        expansions: list[str] = []

        for word in words:
            if len(word) < 4:
                continue
            synsets = wordnet.synsets(word)
            seen = {word}
            count = 0
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ").lower()
                    if name not in seen:
                        seen.add(name)
                        expansions.append(name)
                        count += 1
                        if count >= max_synonyms_per_word:
                            break
                if count >= max_synonyms_per_word:
                    break

        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query

    # ── Multi-Turn Context ────────────────────────────────────────────────

    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self._history.append({"role": role, "content": content})
        if len(self._history) > self._max_history * 2:  # user + assistant pairs
            self._history = self._history[-self._max_history * 2:]

    def get_contextualised_query(self, query: str) -> str:
        """
        Prepend relevant recent context to the query for continuity.

        If the query looks like a follow-up (e.g. starts with a pronoun),
        prepend the last Q+A pair.
        """
        follow_up_patterns = r"^(and|also|what about|how about|tell me more|why|but|so)\b"
        if self._history and re.match(follow_up_patterns, query.lower()):
            # Include last exchange as context
            recent = self._history[-2:]  # last Q + A
            context = " ".join(m["content"] for m in recent)
            return f"{context} {query}"
        return query

    def clear_history(self):
        self._history.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    # ── Full Processing Pipeline ──────────────────────────────────────────

    def process(
        self,
        query: str,
        expand: bool = True,
        contextualise: bool = True,
    ) -> dict:
        """
        Run the full query processing pipeline.

        Returns
        -------
        dict with keys:
            - original: the raw input
            - normalized: cleaned query
            - expanded: query with synonyms
            - question_type: classified question type
            - final: the query string to use for retrieval
        """
        normalized = self.normalize(query)

        if contextualise:
            contextualised = self.get_contextualised_query(normalized)
        else:
            contextualised = normalized

        if expand:
            expanded = self.expand_with_synonyms(contextualised)
        else:
            expanded = contextualised

        question_type = classify_question_type(query)

        return {
            "original": query,
            "normalized": normalized,
            "expanded": expanded,
            "question_type": question_type,
            "final": expanded,
        }
