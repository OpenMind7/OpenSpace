"""SkillRanker — BM25 + embedding hybrid ranking for skills.

Provides a two-stage retrieval pipeline for skill selection:
  Stage 1 (BM25): Fast lexical rough-rank over all skills
  Stage 2 (Embedding): Semantic re-rank on BM25 candidates

Embedding strategy:
  - Text = ``name + description + SKILL.md body`` (consistent with MCP
    ``search_skills`` and the clawhub cloud platform)
  - Model: ``qwen/qwen3-embedding-8b`` via OpenRouter API
  - Embeddings are cached in-memory keyed by ``skill_id`` and optionally
    persisted to a pickle file for cross-session reuse

Reused by:
  - ``SkillRegistry.select_skills_with_llm`` — pre-filter before LLM selection
  - ``mcp_server.search_skills`` — BM25 stage of the MCP search tool
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openspace.utils.logging import Logger

logger = Logger.get_logger(__name__)

# Embedding model — must match clawhub platform for vector-space compatibility
SKILL_EMBEDDING_MODEL = "openai/text-embedding-3-small"
SKILL_EMBEDDING_MAX_CHARS = 12_000

# Pre-filter threshold: when local skills exceed this count, BM25 pre-filter
# is activated before LLM selection.  Below this, all skills go directly to LLM.
PREFILTER_THRESHOLD = 10

# How many candidates to keep after BM25 rough-rank (before embedding re-rank)
BM25_CANDIDATES_MULTIPLIER = 3  # top_k * 3

# Cache version — increment when format changes
_CACHE_VERSION = 1


@dataclass
class SkillCandidate:
    """Lightweight skill representation for ranking."""
    skill_id: str
    name: str
    description: str
    body: str = ""             # SKILL.md body (frontmatter stripped)
    source: str = "local"      # "local" | "cloud"
    # Internal ranking fields
    embedding: Optional[List[float]] = None
    embedding_text: str = ""   # text used to compute embedding
    score: float = 0.0
    bm25_score: float = 0.0
    vector_score: float = 0.0
    # Pass-through metadata (for MCP search results)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillRanker:
    """Hybrid BM25 + embedding ranker for skills.

    Usage::

        ranker = SkillRanker()
        candidates = [SkillCandidate(skill_id=..., name=..., description=..., body=...)]
        ranked = ranker.hybrid_rank(query, candidates, top_k=10)
    """

    def __init__(
        self,
        *,
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
        enable_cross_encoder: bool = False,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_top_k: int = 5,
        rerank_cache_max_entries: int = 1024,
        rerank_cache_ttl_seconds: int = 3600,
    ) -> None:
        # Embedding cache: skill_id → List[float]
        self._embedding_cache: Dict[str, List[float]] = {}
        self._enable_cache = enable_cache
        # Cross-encoder (Stage 3) — lazy-loaded
        self._enable_cross_encoder = enable_cross_encoder
        self._cross_encoder_model_name = cross_encoder_model
        self._cross_encoder_top_k = cross_encoder_top_k
        self._cross_encoder: Any = None  # loaded on first use
        # Rerank cache: sha256[:16] → (cached_at, ordered List[skill_id])
        self._rerank_cache: Dict[str, Tuple[float, List[str]]] = {}
        self._rerank_cache_max_entries = rerank_cache_max_entries
        self._rerank_cache_ttl_seconds = rerank_cache_ttl_seconds

        if cache_dir is None:
            try:
                from openspace.config.constants import PROJECT_ROOT
                cache_dir = PROJECT_ROOT / ".openspace" / "skill_embedding_cache"
            except Exception:
                cache_dir = Path(".openspace") / "skill_embedding_cache"
        self._cache_dir = Path(cache_dir)

        if self._enable_cache:
            self._load_cache()

    def hybrid_rank(
        self,
        query: str,
        candidates: List[SkillCandidate],
        top_k: int = 10,
    ) -> List[SkillCandidate]:
        """BM25 rough-rank → embedding re-rank → return top_k.

        Falls back gracefully:
          - No BM25 lib → simple token overlap
          - No embedding API key → BM25-only
          - Both fail → return first top_k candidates
        """
        if not candidates or not query.strip():
            return candidates[:top_k]

        # Stage 1: BM25 rough-rank
        bm25_top = self._bm25_rank(query, candidates, top_k * BM25_CANDIDATES_MULTIPLIER)
        if not bm25_top:
            # BM25 found nothing — try embedding on all candidates
            emb_results = self._embedding_rank(query, candidates, top_k)
            stage2 = emb_results if emb_results else candidates[:top_k]
            return self.rerank(query, stage2, top_k=top_k) if self._enable_cross_encoder else stage2

        # Stage 2: Embedding re-rank on BM25 candidates
        emb_results = self._embedding_rank(query, bm25_top, top_k)
        if emb_results:
            return self.rerank(query, emb_results, top_k=top_k) if self._enable_cross_encoder else emb_results

        # Embedding unavailable — return BM25 results
        logger.debug("Embedding unavailable, using BM25-only results")
        stage2 = bm25_top[:top_k]
        return self.rerank(query, stage2, top_k=top_k) if self._enable_cross_encoder else stage2

    def rerank(
        self,
        query: str,
        candidates: List[SkillCandidate],
        top_k: Optional[int] = None,
    ) -> List[SkillCandidate]:
        """Stage 3: Cross-encoder reranking over Stage 2 candidates.

        Scores each (query, skill_text) pair using a cross-encoder model.
        Results are cached by (query × candidate_set) hash.

        Falls back to returning candidates[:top_k] if the model is unavailable.
        """
        if not candidates or not self._enable_cross_encoder:
            return candidates
        k = top_k if top_k is not None else self._cross_encoder_top_k

        import hashlib
        cache_key = hashlib.sha256(
            f"{query}|{','.join(sorted(c.skill_id for c in candidates))}".encode()
        ).hexdigest()[:16]
        cached = self._rerank_cache.get(cache_key)
        if cached:
            cached_at, ordered_ids = cached
            if time.monotonic() - cached_at <= self._rerank_cache_ttl_seconds:
                id_to_c = {c.skill_id: c for c in candidates}
                result = [id_to_c[sid] for sid in ordered_ids if sid in id_to_c]
                return result[:k]
            self._rerank_cache.pop(cache_key, None)

        ce = self._get_cross_encoder()
        if ce is None:
            return candidates[:k]

        pairs = [
            (query, f"{c.name}: {c.description}\n{c.body[:500]}")
            for c in candidates
        ]
        try:
            scores = ce.predict(pairs)
        except Exception as exc:
            logger.warning("Cross-encoder prediction failed: %s", exc)
            return candidates[:k]

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        result = [c for c, _ in ranked]
        self._rerank_cache[cache_key] = (time.monotonic(), [c.skill_id for c in result])
        while len(self._rerank_cache) > self._rerank_cache_max_entries:
            self._rerank_cache.pop(next(iter(self._rerank_cache)))
        return result[:k]

    def _get_cross_encoder(self) -> Any:
        """Lazy-load the cross-encoder model. Returns None if unavailable."""
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._cross_encoder = CrossEncoder(self._cross_encoder_model_name)
            logger.info("Cross-encoder loaded: %s", self._cross_encoder_model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; cross-encoder disabled. "
                "Install with: pip install sentence-transformers"
            )
            self._enable_cross_encoder = False
        except Exception as exc:
            logger.warning("Failed to load cross-encoder model: %s", exc)
            self._enable_cross_encoder = False
        return self._cross_encoder

    def bm25_only(
        self,
        query: str,
        candidates: List[SkillCandidate],
        top_k: int = 30,
    ) -> List[SkillCandidate]:
        """BM25-only ranking (for MCP search Phase 1)."""
        return self._bm25_rank(query, candidates, top_k)

    def embedding_only(
        self,
        query: str,
        candidates: List[SkillCandidate],
        top_k: int = 10,
    ) -> List[SkillCandidate]:
        """Embedding-only ranking."""
        return self._embedding_rank(query, candidates, top_k)

    def get_or_compute_embedding(
        self, candidate: SkillCandidate,
    ) -> Optional[List[float]]:
        """Get embedding from cache or compute it.

        Returns None if embedding cannot be generated.
        """
        # Already has embedding (e.g. cloud pre-computed)
        if candidate.embedding:
            return candidate.embedding

        # Check cache
        cached = self._embedding_cache.get(candidate.skill_id)
        if cached:
            candidate.embedding = cached
            return cached

        # Compute
        text = self._build_embedding_text(candidate)
        emb = self._generate_embedding(text)
        if emb:
            candidate.embedding = emb
            self._embedding_cache[candidate.skill_id] = emb
            self._save_cache()
        return emb

    def invalidate_cache(self, skill_id: str) -> None:
        """Remove a skill's cached embedding (e.g. after evolution)."""
        self._embedding_cache.pop(skill_id, None)
        self._save_cache()

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._embedding_cache.clear()
        self._save_cache()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for BM25."""
        tokens = re.split(r"[^\w]+", text.lower())
        return [t for t in tokens if t]

    def _bm25_rank(
        self,
        query: str,
        candidates: List[SkillCandidate],
        top_k: int,
    ) -> List[SkillCandidate]:
        """Rank candidates using BM25."""
        if not candidates:
            return []

        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError:
            BM25Okapi = None

        # Build corpus: name + description + truncated body for richer matching
        corpus_tokens = []
        for c in candidates:
            text = f"{c.name} {c.description}"
            if c.body:
                text += f" {c.body[:2000]}"  # include body for BM25 but cap length
            corpus_tokens.append(self._tokenize(text))

        query_tokens = self._tokenize(query)

        if BM25Okapi and corpus_tokens:
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
            for c, s in zip(candidates, scores):
                c.bm25_score = float(s)
        else:
            # Fallback: simple token overlap
            q_set = set(query_tokens)
            for c, toks in zip(candidates, corpus_tokens):
                if not toks or not q_set:
                    c.bm25_score = 0.0
                else:
                    overlap = q_set.intersection(toks)
                    c.bm25_score = len(overlap) / len(q_set)

        # Sort and filter
        ranked = sorted(candidates, key=lambda c: c.bm25_score, reverse=True)

        # If all scores are 0 (no match), return all candidates (let embedding decide)
        if all(c.bm25_score == 0.0 for c in ranked):
            logger.debug("BM25 found no matches, passing all candidates to embedding stage")
            return candidates[:top_k]

        return ranked[:top_k]

    @staticmethod
    def _get_openai_api_key() -> Optional[str]:
        """Resolve OpenAI-compatible API key for embedding requests."""
        from openspace.cloud.embedding import resolve_embedding_api
        api_key, _ = resolve_embedding_api()
        return api_key

    @staticmethod
    def _build_embedding_text(candidate: SkillCandidate) -> str:
        """Build text for embedding, consistent with MCP search_skills."""
        if candidate.embedding_text:
            return candidate.embedding_text
        header = "\n".join(filter(None, [candidate.name, candidate.description]))
        raw = "\n\n".join(filter(None, [header, candidate.body]))
        if len(raw) > SKILL_EMBEDDING_MAX_CHARS:
            raw = raw[:SKILL_EMBEDDING_MAX_CHARS]
        candidate.embedding_text = raw
        return raw

    def _embedding_rank(
        self,
        query: str,
        candidates: List[SkillCandidate],
        top_k: int,
    ) -> List[SkillCandidate]:
        """Rank candidates using embedding cosine similarity."""
        api_key = self._get_openai_api_key()
        if not api_key:
            return []

        # Generate query embedding
        query_emb = self._generate_embedding(query, api_key=api_key)
        if not query_emb:
            return []

        # Ensure all candidates have embeddings
        for c in candidates:
            if not c.embedding:
                cached = self._embedding_cache.get(c.skill_id)
                if cached:
                    c.embedding = cached
                else:
                    text = self._build_embedding_text(c)
                    emb = self._generate_embedding(text, api_key=api_key)
                    if emb:
                        c.embedding = emb
                        self._embedding_cache[c.skill_id] = emb

        # Save newly computed embeddings
        self._save_cache()

        # Score
        for c in candidates:
            if c.embedding:
                c.vector_score = _cosine_similarity(query_emb, c.embedding)
            else:
                c.vector_score = 0.0
            c.score = c.vector_score

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _generate_embedding(
        text: str,
        api_key: Optional[str] = None,
    ) -> Optional[List[float]]:
        """Generate embedding via OpenAI-compatible API (text-embedding-3-small).

        Delegates credential / base-URL resolution to
        :func:`openspace.cloud.embedding.resolve_embedding_api`.
        """
        from openspace.cloud.embedding import resolve_embedding_api

        resolved_key, base_url = resolve_embedding_api()
        if not api_key:
            api_key = resolved_key
        if not api_key:
            return None

        import urllib.request

        body = json.dumps({
            "model": SKILL_EMBEDDING_MODEL,
            "input": text,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/embeddings",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        import time
        last_err = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("data", [{}])[0].get("embedding")
            except Exception as e:
                last_err = e
                if attempt < 2:
                    delay = 2 * (attempt + 1)
                    logger.debug("Embedding request failed (attempt %d/3), retrying in %ds: %s", attempt + 1, delay, e)
                    time.sleep(delay)
        logger.warning("Skill embedding generation failed after 3 attempts: %s", last_err)
        return None

    def _cache_file(self) -> Path:
        return self._cache_dir / f"skill_embeddings_v{_CACHE_VERSION}.pkl"

    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        path = self._cache_file()
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and data.get("version") == _CACHE_VERSION:
                self._embedding_cache = data.get("embeddings", {})
                logger.debug(f"Loaded {len(self._embedding_cache)} skill embeddings from cache")
        except Exception as e:
            logger.warning(f"Failed to load skill embedding cache: {e}")
            self._embedding_cache = {}

    def _save_cache(self) -> None:
        """Persist embedding cache to disk."""
        if not self._enable_cache or not self._embedding_cache:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "version": _CACHE_VERSION,
                "model": SKILL_EMBEDDING_MODEL,
                "last_updated": datetime.now().isoformat(),
                "embeddings": self._embedding_cache,
            }
            with open(self._cache_file(), "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Failed to save skill embedding cache: {e}")

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_skill_embedding_text(
    name: str,
    description: str,
    readme_body: str,
    max_chars: int = SKILL_EMBEDDING_MAX_CHARS,
) -> str:
    """Build text for skill embedding: ``name + description + SKILL.md body``.

    Unified strategy matching MCP search_skills and clawhub platform.
    """
    header = "\n".join(filter(None, [name, description]))
    raw = "\n\n".join(filter(None, [header, readme_body]))
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars]

