"""
Multi-provider LLM client for answer generation.

Supports: Groq, Ollama, OpenAI, Google Gemini.
Wraps the respective APIs, handles prompt engineering, response validation,
streaming, and fallback to retrieved passages when the API is unavailable.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Generator

import httpx

from config.settings import settings
from retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of LLM-based answer generation."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    is_fallback: bool = False


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise, multilingual question-answering assistant powered by a Retrieval-Augmented Generation (RAG) system.

**Your Rules:**
1. Answer the question based ONLY on the provided context passages.
2. If the context does not contain enough information, say so clearly — do NOT guess or hallucinate.
3. Cite which source(s) you used by referencing [Source N].
4. Keep your answers concise and well-structured.
5. If the question is in a language other than English, respond in the same language.
6. If multiple sources agree, synthesise them into a single coherent answer.
"""


def _build_context_block(results: list[RetrievalResult]) -> str:
    """Format retrieval results into a numbered context block for the prompt."""
    parts = []
    for r in results:
        doc = r.document
        parts.append(
            f"[Source {r.rank}] (score: {r.score:.3f})\\n"
            f"Question: {doc.question}\\n"
            f"Answer: {doc.answer_text[:1000]}\\n"
        )
    return "\\n---\\n".join(parts)


def _build_source_list(results: list[RetrievalResult]) -> list[dict]:
    """Build a structured source list for the response."""
    return [
        {
            "rank": r.rank,
            "score": round(r.score, 4),
            "question": r.document.question,
            "answer_preview": r.document.answer_text[:200],
            "metadata": r.document.metadata,
        }
        for r in results
    ]


# ── Base LLM Client ──────────────────────────────────────────────────────────

class BaseLLMClient:
    """Abstract base class for all LLM providers."""

    def __init__(self):
        self.model = settings.llm_model

    def generate(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        conversation_history: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        
        if not retrieval_results:
            return GenerationResult(
                answer="I couldn't find any relevant information to answer your question.",
                confidence=0.0,
                is_fallback=True,
            )

        sources = _build_source_list(retrieval_results)

        try:
            return self._generate_with_llm(
                query=query,
                retrieval_results=retrieval_results,
                sources=sources,
                conversation_history=conversation_history,
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
        except Exception as e:
            logger.error("%s generation failed: %s — falling back", self.__class__.__name__, e)
            return self._fallback_answer(query, retrieval_results, sources)

    def _generate_with_llm(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        sources: list[dict],
        conversation_history: Optional[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        raise NotImplementedError

    def generate_stream(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        raise NotImplementedError

    @staticmethod
    def _fallback_answer(
        query: str,
        retrieval_results: list[RetrievalResult],
        sources: list[dict],
    ) -> GenerationResult:
        """Return top retrieved passage as the answer (no LLM)."""
        top = retrieval_results[0]
        answer = (
            f"**[Fallback — LLM unavailable]**\\n\\n"
            f"Based on the most relevant passage (score: {top.score:.3f}):\\n\\n"
            f"{top.document.answer_text[:2000]}"
        )
        return GenerationResult(
            answer=answer,
            sources=sources,
            confidence=top.score,
            latency_ms=0.0,
            model="fallback",
            is_fallback=True,
        )

    def _calculate_confidence(self, retrieval_results: list[RetrievalResult]) -> float:
        avg_score = sum(r.score for r in retrieval_results) / len(retrieval_results)
        return min(avg_score, 1.0)

    def _build_messages(self, query: str, retrieval_results: list[RetrievalResult], conversation_history: Optional[list[dict]]):
        context_block = _build_context_block(retrieval_results)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history[-6:])  # last 3 exchanges
        user_message = (
            f"**Context:**\\n{context_block}\\n\\n"
            f"**Question:** {query}\\n\\n"
            f"Please answer based on the context above."
        )
        messages.append({"role": "user", "content": user_message})
        return messages


# ── Groq Client ──────────────────────────────────────────────────────────────

class GroqLLMClient(BaseLLMClient):

    def __init__(self):
        super().__init__()
        self.api_key = settings.groq_api_key
        self._client = None
        if self.api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info("Groq client initialised (model=%s)", self.model)
            except ImportError:
                logger.warning("groq package not installed")

    def _generate_with_llm(self, query, retrieval_results, sources, conversation_history, temperature, max_tokens) -> GenerationResult:
        if not self._client:
            raise RuntimeError("Groq client not configured")
        
        t0 = time.perf_counter()
        messages = self._build_messages(query, retrieval_results, conversation_history)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        latency = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            answer=answer, sources=sources, confidence=self._calculate_confidence(retrieval_results),
            latency_ms=latency, model=self.model
        )


# ── Ollama Client ────────────────────────────────────────────────────────────

class OllamaLLMClient(BaseLLMClient):

    def __init__(self):
        super().__init__()
        self.base_url = settings.ollama_base_url.rstrip('/')
        logger.info("Ollama client initialised (model=%s, url=%s)", self.model, self.base_url)

    def _generate_with_llm(self, query, retrieval_results, sources, conversation_history, temperature, max_tokens) -> GenerationResult:
        t0 = time.perf_counter()
        messages = self._build_messages(query, retrieval_results, conversation_history)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        with httpx.Client() as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            
        answer = data.get("message", {}).get("content", "").strip()
        latency = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            answer=answer, sources=sources, confidence=self._calculate_confidence(retrieval_results),
            latency_ms=latency, model=self.model
        )


# ── OpenAI Client ────────────────────────────────────────────────────────────

class OpenAILLMClient(BaseLLMClient):

    def __init__(self):
        super().__init__()
        self.api_key = settings.openai_api_key
        logger.info("OpenAI client initialised (model=%s)", self.model)

    def _generate_with_llm(self, query, retrieval_results, sources, conversation_history, temperature, max_tokens) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("OpenAI API key not configured")
            
        t0 = time.perf_counter()
        messages = self._build_messages(query, retrieval_results, conversation_history)
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        with httpx.Client() as client:
            response = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            
        answer = data["choices"][0]["message"]["content"].strip()
        latency = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            answer=answer, sources=sources, confidence=self._calculate_confidence(retrieval_results),
            latency_ms=latency, model=self.model
        )


# ── Google Gemini Client ─────────────────────────────────────────────────────

class GoogleLLMClient(BaseLLMClient):

    def __init__(self):
        super().__init__()
        self.api_key = settings.google_api_key
        logger.info("Google client initialised (model=%s)", self.model)

    def _generate_with_llm(self, query, retrieval_results, sources, conversation_history, temperature, max_tokens) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("Google API key not configured")
            
        t0 = time.perf_counter()
        messages = self._build_messages(query, retrieval_results, conversation_history)
        
        # Convert standard messages format to Gemini format
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        with httpx.Client() as client:
            response = client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            
        answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        latency = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            answer=answer, sources=sources, confidence=self._calculate_confidence(retrieval_results),
            latency_ms=latency, model=self.model
        )


# ── Factory ──────────────────────────────────────────────────────────────────

def get_llm_client() -> BaseLLMClient:
    """Instantiate the appropriate LLM client based on settings."""
    provider = settings.llm_provider.lower().strip()
    
    if provider == "ollama":
        return OllamaLLMClient()
    elif provider == "openai":
        return OpenAILLMClient()
    elif provider == "google":
        return GoogleLLMClient()
    elif provider == "groq":
        return GroqLLMClient()
    else:
        logger.warning("Unknown LLM provider '%s', defaulting to Ollama client", provider)
        return OllamaLLMClient()
