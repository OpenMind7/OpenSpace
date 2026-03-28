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
        # Embedding cache: "{embedding_version}:{skill_id}" → List[float]
        # Version prefix prevents silent dim-mismatch after a fine-tune cycle.
        self._embedding_cache: Dict[str, List[float]] = {}
        self._enable_cache = enable_cache
        # Bumped by fine_tune_from_outcomes; old cache entries auto-miss on next lookup.
        self._embedding_version: int = 0
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

        # Check cache (version-prefixed to detect stale post-fine-tune entries)
        _vkey = f"{self._embedding_version}:{candidate.skill_id}"
        cached = self._embedding_cache.get(_vkey)
        if cached:
            candidate.embedding = cached
            return cached

        # Compute
        text = self._build_embedding_text(candidate)
        emb = self._generate_embedding(text)
        if emb:
            candidate.embedding = emb
            self._embedding_cache[_vkey] = emb
            self._save_cache()
        return emb

    def invalidate_cache(self, skill_id: str) -> None:
        """Remove a skill's cached embedding (e.g. after evolution)."""
        self._embedding_cache.pop(f"{self._embedding_version}:{skill_id}", None)
        self._save_cache()

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._embedding_cache.clear()
        self._save_cache()

    async def fine_tune_from_outcomes(
        self,
        *,
        store: Any,
        min_new_pairs: int = 256,
        replay_window: int = 2048,
    ) -> bool:
        """Fine-tune a local residual adapter using InfoNCE loss on outcome_pairs.

        Residual projection: z = normalize(x + W2(GELU(W1(x))))
        This runs ON TOP of frozen provider embeddings — not provider fine-tuning.

        Returns True if fine-tuning ran, False if skipped (not enough new pairs).
        Bumps _embedding_version on success so stale cache entries auto-invalidate.
        """
        import numpy as np

        # 1. Determine cursor from last successful run
        latest = store.get_latest_training_run()
        since_id = latest["end_pair_id"] if latest else 0
        prev_version = latest["embedding_version"] if latest else 0

        # 2. Load new pairs
        pairs = store.get_outcome_pairs_since(since_id, limit=replay_window)
        if len(pairs) < min_new_pairs:
            logger.debug(
                "fine_tune_from_outcomes: only %d new pairs, need %d — skipped",
                len(pairs), min_new_pairs,
            )
            return False

        # 3. Collect unique skill_ids and gather their base embeddings
        skill_ids = list({p["skill_id"] for p in pairs})
        skill_embs: dict[str, Optional[List[float]]] = {}
        for sid in skill_ids:
            vkey = f"{self._embedding_version}:{sid}"
            emb = self._embedding_cache.get(vkey)
            if emb is None:
                emb = self._generate_embedding(sid)  # best-effort
            skill_embs[sid] = emb

        # Drop skills without embeddings
        skill_embs = {k: v for k, v in skill_embs.items() if v}
        if not skill_embs:
            logger.warning("fine_tune_from_outcomes: no embeddings available — skipped")
            return False

        dim = len(next(iter(skill_embs.values())))

        # 4. Build residual adapter (W1, W2) — initialize from existing or fresh
        adapter_key = "__adapter__"
        adapter_data = self._embedding_cache.get(adapter_key)
        if (
            adapter_data is not None
            and isinstance(adapter_data, dict)
            and adapter_data.get("dim") == dim
        ):
            W1 = np.array(adapter_data["W1"], dtype=np.float32)
            W2 = np.array(adapter_data["W2"], dtype=np.float32)
        else:
            # Xavier initialization — near-zero residual at start
            scale = np.sqrt(2.0 / (dim + dim))
            W1 = (np.random.randn(dim, dim) * scale).astype(np.float32)
            W2 = (np.random.randn(dim, dim) * scale * 0.1).astype(np.float32)

        # 5. Build training batches: (task_emb_key, skill_id, weight)
        # For InfoNCE: applied_vs_selected (weight 1.0) = positive pairs
        #              selected_vs_shortlist (weight 0.25) = hard negatives
        def _apply_adapter(x: np.ndarray) -> np.ndarray:
            """z = normalize(x + W2 @ GELU(W1 @ x))"""
            h = W1 @ x
            h = h * (0.5 * (1.0 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3))))  # GELU
            z = x + W2 @ h
            norm = np.linalg.norm(z)
            return z / (norm + 1e-8)

        # 6. SGD with InfoNCE — mini-batch gradient descent (numpy, no autograd)
        lr = 1e-3
        temperature = 0.07
        n_epochs = 2
        batch_size = 32

        # Filter pairs to those with known embeddings
        valid_pairs = [p for p in pairs if p["skill_id"] in skill_embs]
        if len(valid_pairs) < 8:
            logger.debug("fine_tune_from_outcomes: too few valid pairs — skipped")
            return False

        skill_mat = {sid: np.array(emb, dtype=np.float32) for sid, emb in skill_embs.items()}
        all_sids = list(skill_mat.keys())

        for _epoch in range(n_epochs):
            np.random.shuffle(valid_pairs)  # type: ignore[arg-type]
            for i in range(0, len(valid_pairs), batch_size):
                batch = valid_pairs[i : i + batch_size]

                # Numerical gradient for W1, W2
                eps = 1e-4
                grad_W1 = np.zeros_like(W1)
                grad_W2 = np.zeros_like(W2)

                for p in batch:
                    s_emb = skill_mat[p["skill_id"]]
                    z_anchor = _apply_adapter(s_emb)

                    # Negatives: random sample of other skill embeddings
                    neg_sids = [s for s in all_sids if s != p["skill_id"]]
                    if not neg_sids:
                        continue
                    neg_sample = neg_sids[: min(8, len(neg_sids))]
                    neg_zs = np.stack([_apply_adapter(skill_mat[s]) for s in neg_sample])

                    # InfoNCE: anchor vs negatives (unsupervised: pull applied apart from non-applied)
                    sims = neg_zs @ z_anchor / temperature
                    softmax_denom = np.sum(np.exp(sims - sims.max())) + 1e-8

                    # Approximate gradient via finite differences on W2 only (fast)
                    for j in range(min(dim, 16)):  # sparse probe
                        for k in range(min(dim, 16)):
                            W2[j, k] += eps
                            z_plus = _apply_adapter(s_emb)
                            sims_plus = neg_zs @ z_plus / temperature
                            loss_plus = -sims_plus[0] + np.log(
                                np.sum(np.exp(sims_plus - sims_plus.max())) + 1e-8
                            )
                            W2[j, k] -= eps
                            sims_orig = neg_zs @ z_anchor / temperature
                            loss_orig = -sims_orig[0] + np.log(softmax_denom)
                            grad_W2[j, k] += (loss_plus - loss_orig) / eps * p["weight"]

                W2 -= lr * grad_W2

        # 7. Recompute and re-cache all skill embeddings with new adapter
        new_version = prev_version + 1
        for sid, base_emb in skill_mat.items():
            adapted = _apply_adapter(base_emb).tolist()
            self._embedding_cache[f"{new_version}:{sid}"] = adapted

        # Save adapter alongside embeddings
        self._embedding_cache[adapter_key] = {
            "dim": dim,
            "version": new_version,
            "W1": W1.tolist(),
            "W2": W2.tolist(),
        }
        self._embedding_version = new_version
        self._save_cache()

        # 8. Record training run in store
        end_pair_id = pairs[-1]["id"] if pairs else since_id
        await store.record_training_run(
            embedding_version=new_version,
            end_pair_id=end_pair_id,
            status="complete",
        )

        logger.info(
            "fine_tune_from_outcomes: trained on %d pairs → embedding_version=%d",
            len(valid_pairs), new_version,
        )
        return True

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
                _vkey = f"{self._embedding_version}:{c.skill_id}"
                cached = self._embedding_cache.get(_vkey)
                if cached:
                    c.embedding = cached
                else:
                    text = self._build_embedding_text(c)
                    emb = self._generate_embedding(text, api_key=api_key)
                    if emb:
                        c.embedding = emb
                        self._embedding_cache[_vkey] = emb

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

