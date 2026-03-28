"""Hybrid skill search engine (BM25 + embedding + lexical boost + nexus-kb).

Implements the search pipeline:
  Phase 0: nexus-kb FTS5 search (knowledge base cross-validation)
  Phase 1: BM25 rough-rank over all candidates (with query expansion + body content)
  Phase 2: Vector scoring (embedding cosine similarity)
  Phase 3: RRF fusion (BM25 rank + vector rank) + lexical + nexus boosts
  Phase 4: Deduplication + limit

Wave 2 upgrades (2026-03-28):
  - BM25 body fix: SKILL.md body content included in BM25 ranking
  - RRF fusion: Reciprocal Rank Fusion replaces simple score addition
  - Query expansion: static synonym expansion before BM25 phase

Used by MCP ``search_skills`` tool, ``retrieve_skill`` agent tool,
and potentially other search interfaces.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openspace.cloud")

# Cross-validation boost when a skill matches nexus-kb keywords.
_NEXUS_KB_CROSS_VALIDATION_BOOST = 0.2

# RRF constant k (standard value; higher k = less aggressive rank weighting).
_RRF_K = 60

# Static synonym expansion table for common tech/automation concepts.
# Keys are query tokens; values are additional tokens added to the BM25 query.
_QUERY_SYNONYMS: dict[str, list[str]] = {
    # File/dir operations
    "list":      ["enumerate", "directory", "ls", "show", "get"],
    "read":      ["load", "open", "parse", "fetch", "retrieve"],
    "write":     ["save", "store", "create", "output", "dump"],
    "delete":    ["remove", "rm", "unlink", "erase", "cleanup"],
    "search":    ["find", "grep", "query", "lookup", "scan"],
    "copy":      ["cp", "duplicate", "clone"],
    "move":      ["mv", "rename", "transfer"],
    # Web / network
    "download":  ["fetch", "pull", "get", "http", "request"],
    "upload":    ["push", "send", "post", "transfer"],
    "scrape":    ["crawl", "extract", "parse", "harvest"],
    "request":   ["http", "get", "post", "api", "call"],
    # Code / dev
    "run":       ["execute", "invoke", "call", "exec", "launch"],
    "test":      ["check", "verify", "validate", "assert", "spec"],
    "debug":     ["trace", "diagnose", "inspect", "troubleshoot"],
    "build":     ["compile", "make", "bundle", "package"],
    "deploy":    ["publish", "release", "ship", "push"],
    "install":   ["setup", "configure", "add", "package"],
    # Data
    "parse":     ["decode", "extract", "process", "interpret"],
    "format":    ["convert", "transform", "serialize", "render"],
    "summarize": ["summarise", "condense", "synopsis", "brief"],
    "analyze":   ["analyse", "review", "inspect", "evaluate"],
    "generate":  ["create", "produce", "make", "write", "synthesize"],
    # AI/ML
    "embed":     ["embedding", "vector", "encode"],
    "classify":  ["categorize", "label", "detect", "identify"],
    "predict":   ["infer", "forecast", "estimate"],
}


def _expand_query(query: str) -> str:
    """Expand query with synonyms for BM25 phase.

    Appends synonym tokens for known tech terms so that skills matching
    related vocabulary are not dropped in the BM25 pre-filter.
    Original query tokens are preserved; expansion only adds extras.

    Example:
        "list files in directory" → "list enumerate directory ls show files ..."
    """
    tokens = _WORD_RE.findall(query.lower())
    expansions: list[str] = []
    for tok in tokens:
        syns = _QUERY_SYNONYMS.get(tok)
        if syns:
            expansions.extend(syns)
    if not expansions:
        return query
    return query + " " + " ".join(expansions)


def _check_safety(text: str) -> list[str]:
    """Lazy wrapper — avoids importing skill_engine at module load time."""
    from openspace.skill_engine.skill_utils import check_skill_safety
    return check_skill_safety(text)


def _is_safe(flags: list[str]) -> bool:
    from openspace.skill_engine.skill_utils import is_skill_safe
    return is_skill_safe(flags)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(value: str) -> list[str]:
    return _WORD_RE.findall(value.lower()) if value else []


def _lexical_boost(query_tokens: list[str], name: str, slug: str) -> float:
    """Compute lexical boost score based on exact/prefix token matching."""
    slug_tokens = _tokenize(slug)
    name_tokens = _tokenize(name)
    boost = 0.0

    # Slug exact / prefix
    if slug_tokens and all(
        any(ct == qt for ct in slug_tokens) for qt in query_tokens
    ):
        boost += 1.4
    elif slug_tokens and all(
        any(ct.startswith(qt) for ct in slug_tokens) for qt in query_tokens
    ):
        boost += 0.8

    # Name exact / prefix
    if name_tokens and all(
        any(ct == qt for ct in name_tokens) for qt in query_tokens
    ):
        boost += 1.1
    elif name_tokens and all(
        any(ct.startswith(qt) for ct in name_tokens) for qt in query_tokens
    ):
        boost += 0.6

    return boost


class SkillSearchEngine:
    """Hybrid BM25 + embedding search engine for skills.

    Usage::

        engine = SkillSearchEngine()
        results = engine.search(
            query="weather forecast",
            candidates=candidates,
            query_embedding=[...],  # optional
            limit=20,
        )
    """

    def search(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        *,
        query_embedding: Optional[List[float]] = None,
        nexus_kb_keywords: Optional[set] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Run the full search pipeline on candidates.

        Each candidate dict should have at minimum:
          - ``skill_id``, ``name``, ``description``
          - ``_embedding`` (optional): pre-computed embedding vector
          - ``source``: "openspace-local" | "cloud"

        Args:
            query: Search query text.
            candidates: Candidate dicts to rank.
            query_embedding: Pre-computed query embedding (if available).
            nexus_kb_keywords: Keywords from nexus-kb hits for cross-validation.
            limit: Max results to return.

        Returns:
            Sorted list of result dicts (highest score first).
        """
        q = query.strip()
        if not q or not candidates:
            return []

        query_tokens = _tokenize(q)
        if not query_tokens:
            return []

        # Phase 1: BM25 rough-rank (query-expanded, body-aware)
        filtered, bm25_rank_map = self._bm25_phase(q, candidates, limit)

        # Phase 2+3: RRF fusion (vector rank + BM25 rank) + lexical + nexus boosts
        scored = self._score_phase(
            filtered, query_tokens, query_embedding, nexus_kb_keywords,
            bm25_rank_map=bm25_rank_map,
        )

        # Phase 4: Deduplicate and limit
        return self._dedup_and_limit(scored, limit)

    def _bm25_phase(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        limit: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """BM25 rough-rank to keep top candidates for embedding stage.

        Wave 2 changes:
          - Query is expanded with synonyms before BM25 (improves recall)
          - SkillCandidate body is populated from ``_body`` / ``_embedding_text``
            so skills matching only in SKILL.md body are no longer dropped
          - Returns a bm25_rank_map {skill_id: rank} for RRF fusion

        Returns:
            (filtered_candidates, bm25_rank_map)
        """
        from openspace.skill_engine.skill_ranker import SkillRanker, SkillCandidate

        expanded_query = _expand_query(query)

        ranker = SkillRanker(enable_cache=True)
        bm25_candidates = [
            SkillCandidate(
                skill_id=c.get("skill_id", ""),
                name=c.get("name", ""),
                description=c.get("description", ""),
                # Use SKILL.md body content; fall back to embedding_text (covers all text)
                body=c.get("_body", c.get("_embedding_text", "")),
                metadata=c,
            )
            for c in candidates
        ]
        ranked = ranker.bm25_only(
            expanded_query, bm25_candidates,
            top_k=min(limit * 3, len(candidates)),
        )

        # Preserve BM25 ranking order for RRF fusion
        bm25_rank_map: Dict[str, int] = {sc.skill_id: rank for rank, sc in enumerate(ranked)}
        ranked_ids = set(bm25_rank_map.keys())
        filtered = [c for c in candidates if c.get("skill_id") in ranked_ids]

        # If BM25 found nothing, fall back to all candidates (unranked → rank = index)
        if not filtered:
            return candidates, {c.get("skill_id", ""): i for i, c in enumerate(candidates)}
        return filtered, bm25_rank_map

    def _score_phase(
        self,
        candidates: List[Dict[str, Any]],
        query_tokens: list[str],
        query_embedding: Optional[List[float]],
        nexus_kb_keywords: Optional[set] = None,
        bm25_rank_map: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Compute hybrid score using RRF fusion + lexical + nexus boosts.

        Wave 2 RRF approach:
          1. BM25 rank is taken from bm25_rank_map (preserved from Phase 1)
          2. Vector scores are computed; candidates sorted to get vector rank
          3. RRF = 1/(k + bm25_rank) + 1/(k + vector_rank)
          4. Lexical + nexus boosts applied as additive tiebreakers

        When no embedding is available, BM25-only RRF is used (vector_rank
        defaults to position in input list, which is arbitrary but consistent).
        """
        from openspace.cloud.embedding import cosine_similarity

        bm25_rank_map = bm25_rank_map or {}

        # Step 1: Compute vector scores for all candidates
        vector_scores: list[tuple[int, float]] = []  # (original_index, score)
        raw_entries: list[Dict[str, Any]] = []

        for idx, c in enumerate(candidates):
            name = c.get("name", "")
            slug = c.get("skill_id", name).split("__")[0].replace(":", "-")

            vector_score = 0.0
            if query_embedding:
                skill_emb = c.get("_embedding")
                if skill_emb and isinstance(skill_emb, list):
                    vector_score = cosine_similarity(query_embedding, skill_emb)

            lexical = _lexical_boost(query_tokens, name, slug)

            nexus_boost = 0.0
            if nexus_kb_keywords:
                name_tokens = _tokenize(name)
                desc_tokens = _tokenize(c.get("description", ""))
                candidate_tokens = set(name_tokens + desc_tokens)
                if candidate_tokens & nexus_kb_keywords:
                    nexus_boost = _NEXUS_KB_CROSS_VALIDATION_BOOST

            vector_scores.append((idx, vector_score))
            raw_entries.append({
                "_c": c,
                "_idx": idx,
                "_name": name,
                "_slug": slug,
                "_vector_score": vector_score,
                "_lexical": lexical,
                "_nexus_boost": nexus_boost,
            })

        # Step 2: Derive vector rank (highest vector_score = rank 0)
        sorted_by_vector = sorted(vector_scores, key=lambda x: -x[1])
        vector_rank_map: Dict[int, int] = {
            orig_idx: rank for rank, (orig_idx, _) in enumerate(sorted_by_vector)
        }

        # Step 3: RRF fusion + lexical/nexus boosts
        scored = []
        for e in raw_entries:
            c = e["_c"]
            skill_id = c.get("skill_id", "")
            orig_idx = e["_idx"]

            bm25_rank = bm25_rank_map.get(skill_id, orig_idx)
            vector_rank = vector_rank_map.get(orig_idx, orig_idx)

            rrf = 1.0 / (_RRF_K + bm25_rank) + 1.0 / (_RRF_K + vector_rank)
            final_score = rrf + e["_lexical"] + e["_nexus_boost"]

            entry: Dict[str, Any] = {
                "skill_id": skill_id,
                "name": e["_name"],
                "description": c.get("description", ""),
                "source": c.get("source", ""),
                "score": round(final_score, 6),
            }
            if e["_vector_score"] > 0:
                entry["vector_score"] = round(e["_vector_score"], 4)
            if e["_nexus_boost"] > 0:
                entry["nexus_kb_validated"] = True
            for key in ("path", "visibility", "created_by", "origin", "tags", "quality", "safety_flags"):
                if c.get(key):
                    entry[key] = c[key]
            scored.append(entry)

        scored.sort(key=lambda x: -x["score"])
        return scored

    @staticmethod
    def _dedup_and_limit(
        scored: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Deduplicate by (source, skill_id) with local-preferred tiebreak."""
        seen: dict[str, Dict[str, Any]] = {}
        for item in scored:
            source = item.get("source", "unknown")
            skill_id = item.get("skill_id", item.get("id", ""))
            key = f"{source}:{skill_id}" if skill_id else item["name"]
            if key in seen:
                # Prefer local over cloud, then higher score
                existing = seen[key]
                if source == "local" and existing.get("source") != "local":
                    seen[key] = item
                elif item.get("score", 0) > existing.get("score", 0):
                    seen[key] = item
                continue
            seen[key] = item
        return list(seen.values())[:limit]


def build_local_candidates(
    skills: list,
    store: Any = None,
) -> List[Dict[str, Any]]:
    """Build search candidate dicts from SkillRegistry skills.

    Args:
        skills: List of ``SkillMeta`` from ``registry.list_skills()``.
        store: Optional ``SkillStore`` instance for quality data enrichment.

    Returns:
        List of candidate dicts ready for ``SkillSearchEngine.search()``.
    """
    from openspace.cloud.embedding import build_skill_embedding_text

    candidates: List[Dict[str, Any]] = []
    for s in skills:
        # Read SKILL.md body
        readme_body = ""
        try:
            raw = s.path.read_text(encoding="utf-8")
            m = re.match(r"^---\n.*?\n---\n?", raw, re.DOTALL)
            readme_body = raw[m.end():].strip() if m else raw
        except Exception:
            pass

        embedding_text = build_skill_embedding_text(s.name, s.description, readme_body)

        # Safety check
        flags = _check_safety(embedding_text)
        if not _is_safe(flags):
            logger.info(f"BLOCKED local skill {s.skill_id} — {flags}")
            continue

        candidates.append({
            "skill_id": s.skill_id,
            "name": s.name,
            "description": s.description,
            "source": "openspace-local",
            "path": str(s.path),
            "is_local": True,
            "safety_flags": flags if flags else None,
            "_embedding_text": embedding_text,
            # Wave 2: expose body separately so BM25 can index SKILL.md content
            # without double-counting name/description tokens.
            "_body": readme_body,
        })

    # Enrich with quality data
    if store and candidates:
        try:
            all_records = store.load_all(active_only=True)
            for c in candidates:
                rec = all_records.get(c["skill_id"])
                if rec:
                    c["quality"] = {
                        "total_selections": rec.total_selections,
                        "completion_rate": round(rec.completion_rate, 3),
                        "effective_rate": round(rec.effective_rate, 3),
                    }
                    c["tags"] = rec.tags
        except Exception as e:
            logger.warning(f"Quality lookup failed: {e}")

    return candidates


def build_cloud_candidates(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build search candidate dicts from cloud metadata items.

    Args:
        items: Items from ``OpenSpaceClient.fetch_metadata()``.

    Returns:
        List of candidate dicts (with safety filtering applied).
    """
    candidates: List[Dict[str, Any]] = []
    for item in items:
        name = item.get("name", "")
        desc = item.get("description", "")
        tags = item.get("tags", [])
        safety_text = f"{name}\n{desc}\n{' '.join(tags)}"
        flags = _check_safety(safety_text)
        if not _is_safe(flags):
            continue

        c_entry: Dict[str, Any] = {
            "skill_id": item.get("record_id", ""),
            "name": name,
            "description": desc,
            "source": "cloud",
            "visibility": item.get("visibility", "public"),
            "is_local": False,
            "created_by": item.get("created_by", ""),
            "origin": item.get("origin", ""),
            "tags": tags,
            "safety_flags": flags if flags else None,
        }
        # Carry pre-computed embedding
        platform_emb = item.get("embedding")
        if platform_emb and isinstance(platform_emb, list):
            c_entry["_embedding"] = platform_emb
        candidates.append(c_entry)

    return candidates


async def hybrid_search_skills(
    query: str,
    local_skills: list = None,
    store: Any = None,
    source: str = "all",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Shared cloud+local skill search with graceful fallback.

    Builds candidates, generates embeddings, runs ``SkillSearchEngine``.
    Cloud is attempted when *source* includes it; failures are silently
    skipped so the caller always gets local results at minimum.

    Phase 0 (nexus-kb) runs before BM25 to provide cross-validation
    keywords — skills matching nexus-kb knowledge get a score boost.

    Args:
        query: Free-text search query.
        local_skills: ``SkillMeta`` list (from ``registry.list_skills()``).
        store: Optional ``SkillStore`` for quality enrichment.
        source: ``"all"`` | ``"local"`` | ``"cloud"``.
        limit: Maximum results.

    Returns:
        Ranked result dicts (same format as ``SkillSearchEngine.search()``).
    """
    from openspace.cloud.embedding import generate_embedding

    q = query.strip()
    if not q:
        return []

    # Phase 0: nexus-kb knowledge search for cross-validation keywords.
    nexus_kb_keywords: Optional[set] = None
    nexus_kb_hits: List[Dict[str, Any]] = []
    try:
        from openspace.nexus_kb_bridge import search as nexus_search
        from openspace.nexus_kb_bridge import extract_keywords

        nexus_kb_hits = await asyncio.to_thread(nexus_search, q, limit=8)
        if nexus_kb_hits:
            nexus_kb_keywords = extract_keywords(nexus_kb_hits)
            logger.info(
                "nexus-kb Layer 0: %d hits, %d keywords extracted",
                len(nexus_kb_hits),
                len(nexus_kb_keywords) if nexus_kb_keywords else 0,
            )
    except Exception as e:
        logger.debug("nexus-kb Layer 0 unavailable (non-fatal): %s", e)

    candidates: List[Dict[str, Any]] = []

    if source in ("all", "local") and local_skills:
        candidates.extend(build_local_candidates(local_skills, store))

    if source in ("all", "cloud"):
        try:
            from openspace.cloud.auth import get_openspace_auth
            from openspace.cloud.client import OpenSpaceClient

            auth_headers, api_base = get_openspace_auth()
            if auth_headers:
                client = OpenSpaceClient(auth_headers, api_base)
                try:
                    from openspace.cloud.embedding import resolve_embedding_api
                    has_emb = bool(resolve_embedding_api()[0])
                except Exception:
                    has_emb = False
                items = await asyncio.to_thread(
                    client.fetch_metadata, include_embedding=has_emb, limit=200,
                )
                candidates.extend(build_cloud_candidates(items))
        except Exception as e:
            logger.warning(f"hybrid_search_skills: cloud unavailable: {e}")

    if not candidates:
        return []

    # query embedding (optional — key/URL resolved inside generate_embedding)
    query_embedding: Optional[List[float]] = None
    try:
        query_embedding = await asyncio.to_thread(generate_embedding, q)
        if query_embedding:
            for c in candidates:
                if not c.get("_embedding") and c.get("_embedding_text"):
                    emb = await asyncio.to_thread(
                        generate_embedding, c["_embedding_text"],
                    )
                    if emb:
                        c["_embedding"] = emb
    except Exception:
        pass

    engine = SkillSearchEngine()
    return engine.search(
        q, candidates,
        query_embedding=query_embedding,
        nexus_kb_keywords=nexus_kb_keywords,
        limit=limit,
    )

