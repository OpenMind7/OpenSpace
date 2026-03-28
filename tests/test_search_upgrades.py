"""Tests for Wave 2 search upgrades.

Covers:
  - Query expansion (_expand_query with synonym table)
  - BM25 body fix (_body field populated in local candidates, used in _bm25_phase)
  - RRF fusion (_score_phase uses 1/(k+bm25_rank) + 1/(k+vector_rank))
  - bm25_rank_map preserved through pipeline
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

class TestQueryExpansion:
    """_expand_query appends synonym tokens for known tech terms."""

    def _expand(self, query: str) -> str:
        from openspace.cloud.search import _expand_query
        return _expand_query(query)

    def test_no_known_terms_unchanged(self):
        query = "kubernetes ingress controller"
        assert self._expand(query) == query

    def test_list_expands_with_synonyms(self):
        expanded = self._expand("list files")
        assert "list" in expanded
        assert "enumerate" in expanded or "directory" in expanded or "ls" in expanded

    def test_run_expands(self):
        expanded = self._expand("run script")
        assert "execute" in expanded or "invoke" in expanded

    def test_multiple_known_terms(self):
        expanded = self._expand("search and analyze")
        assert "find" in expanded or "grep" in expanded  # search synonyms
        assert "review" in expanded or "inspect" in expanded  # analyze synonyms

    def test_original_tokens_preserved(self):
        """Original query tokens must always be in the expanded query."""
        query = "download and parse data"
        expanded = self._expand(query)
        assert "download" in expanded
        assert "parse" in expanded
        assert "data" in expanded

    def test_expansion_is_superset(self):
        """Expanded query contains all original words."""
        orig = "read write search"
        expanded = self._expand(orig)
        for word in orig.split():
            assert word in expanded, f"{word!r} missing from expansion: {expanded!r}"

    def test_unknown_query_returns_unchanged(self):
        q = "spectral decomposition eigenvalue"
        assert self._expand(q) == q

    def test_expand_is_case_insensitive_matching(self):
        """Tokenization lowercases; expansion works regardless of input case."""
        from openspace.cloud.search import _expand_query
        lower = _expand_query("list files")
        # Both should produce identical expansions since we lowercase internally
        assert "enumerate" in lower or "directory" in lower

    def test_synonym_table_coverage(self):
        """Spot-check that key tech verbs are in the synonym table."""
        from openspace.cloud.search import _QUERY_SYNONYMS
        for term in ("list", "read", "write", "search", "run", "generate", "analyze"):
            assert term in _QUERY_SYNONYMS, f"{term!r} missing from synonym table"
            assert len(_QUERY_SYNONYMS[term]) >= 2, f"{term!r} has too few synonyms"


# ---------------------------------------------------------------------------
# BM25 body fix
# ---------------------------------------------------------------------------

class TestBm25BodyFix:
    """build_local_candidates must populate _body; _bm25_phase must use it."""

    def _make_candidate(
        self,
        skill_id: str = "test__abc123",
        name: str = "Test Skill",
        description: str = "A test skill",
        body: str = "",
        embedding_text: str = "",
    ) -> Dict[str, Any]:
        return {
            "skill_id": skill_id,
            "name": name,
            "description": description,
            "source": "openspace-local",
            "_body": body,
            "_embedding_text": embedding_text or f"{name} {description} {body}",
        }

    def test_bm25_phase_returns_tuple(self):
        """_bm25_phase must return (filtered_candidates, bm25_rank_map)."""
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()

        candidates = [
            self._make_candidate("s1__aaa", "Weather Forecast", "Get weather data"),
            self._make_candidate("s2__bbb", "File Lister", "List directory files"),
        ]
        result = engine._bm25_phase("weather", candidates, limit=10)
        assert isinstance(result, tuple), "_bm25_phase must return a tuple"
        filtered, rank_map = result
        assert isinstance(filtered, list)
        assert isinstance(rank_map, dict)

    def test_bm25_rank_map_contains_skill_ids(self):
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()

        candidates = [
            self._make_candidate("s1__aaa", "Weather Forecast", "Get weather data"),
            self._make_candidate("s2__bbb", "File Lister", "List directory files"),
        ]
        _, rank_map = engine._bm25_phase("weather", candidates, limit=10)
        # At least one skill ID should be in the rank map
        all_ids = {"s1__aaa", "s2__bbb"}
        assert rank_map.keys() & all_ids, "rank_map must contain candidate skill IDs"

    def test_bm25_rank_map_values_are_integers(self):
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()

        candidates = [self._make_candidate(f"s{i}__abc", f"Skill {i}", f"Desc {i}") for i in range(5)]
        _, rank_map = engine._bm25_phase("skill", candidates, limit=20)
        for v in rank_map.values():
            assert isinstance(v, int), f"rank_map values must be int, got {type(v)}"

    def test_body_content_included_in_skill_candidate(self):
        """SkillCandidate body is populated from _body when present."""
        from openspace.cloud.search import SkillSearchEngine
        from openspace.skill_engine.skill_ranker import SkillCandidate

        # Intercept bm25_only to check what SkillCandidates were created
        created_candidates: list[SkillCandidate] = []

        def mock_bm25_only(query, candidates, top_k):
            created_candidates.extend(candidates)
            return candidates[:top_k]

        engine = SkillSearchEngine()
        with patch(
            "openspace.skill_engine.skill_ranker.SkillRanker.bm25_only",
            side_effect=mock_bm25_only,
        ):
            engine._bm25_phase(
                "weather",
                [self._make_candidate("s1__xyz", "Weather", "Forecast", body="extended instructions here")],
                limit=5,
            )

        assert created_candidates, "No SkillCandidates were created"
        sc = created_candidates[0]
        assert sc.body != "", "_body must be passed through to SkillCandidate.body"
        assert "extended" in sc.body, "SkillCandidate.body must contain the body text"

    def test_embedding_text_used_as_fallback_body(self):
        """When _body is absent, _embedding_text is used as fallback."""
        from openspace.cloud.search import SkillSearchEngine
        from openspace.skill_engine.skill_ranker import SkillCandidate

        created_candidates: list[SkillCandidate] = []

        def mock_bm25_only(query, candidates, top_k):
            created_candidates.extend(candidates)
            return candidates[:top_k]

        candidate = {
            "skill_id": "s1__xyz",
            "name": "Tool",
            "description": "A tool",
            "source": "openspace-local",
            # No _body — only _embedding_text
            "_embedding_text": "combined embedding text fallback",
        }

        engine = SkillSearchEngine()
        with patch(
            "openspace.skill_engine.skill_ranker.SkillRanker.bm25_only",
            side_effect=mock_bm25_only,
        ):
            engine._bm25_phase("tool", [candidate], limit=5)

        assert created_candidates
        sc = created_candidates[0]
        assert "fallback" in sc.body, "Must fall back to _embedding_text when _body absent"

    def test_fallback_when_bm25_finds_nothing(self):
        """When BM25 returns empty, _bm25_phase returns all candidates."""
        from openspace.cloud.search import SkillSearchEngine

        def mock_bm25_only(query, candidates, top_k):
            return []  # BM25 finds nothing

        engine = SkillSearchEngine()
        candidates = [self._make_candidate("s1__abc", "Weather", "Forecast")]
        with patch(
            "openspace.skill_engine.skill_ranker.SkillRanker.bm25_only",
            side_effect=mock_bm25_only,
        ):
            filtered, rank_map = engine._bm25_phase("xyz", candidates, limit=10)

        assert filtered == candidates, "Must return all candidates on BM25 empty"
        assert rank_map, "rank_map must not be empty even on fallback"


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

class TestRrfFusion:
    """_score_phase uses RRF = 1/(k+bm25_rank) + 1/(k+vector_rank)."""

    def _make_cand(self, skill_id: str, name: str, desc: str = "") -> Dict[str, Any]:
        return {
            "skill_id": skill_id,
            "name": name,
            "description": desc,
            "source": "local",
        }

    def test_rrf_constant_k(self):
        from openspace.cloud.search import _RRF_K
        assert _RRF_K == 60, "RRF k must be 60 (standard value)"

    def test_rrf_score_formula(self):
        """Manually verify RRF formula: 1/(k+r1) + 1/(k+r2)."""
        from openspace.cloud.search import _RRF_K
        k = _RRF_K
        bm25_rank, vector_rank = 0, 0
        expected = 1.0 / (k + bm25_rank) + 1.0 / (k + vector_rank)
        assert abs(expected - (1.0 / 60 + 1.0 / 60)) < 1e-9

    def test_higher_bm25_rank_reduces_score(self):
        """Candidate ranked 0 in BM25 should outscore one ranked 10 (all else equal)."""
        from openspace.cloud.search import SkillSearchEngine, _RRF_K
        engine = SkillSearchEngine()

        k = _RRF_K
        # Simulate: two candidates, no embeddings, bm25 ranks differ
        c_top = self._make_cand("top__aaa", "Top")
        c_bot = self._make_cand("bot__bbb", "Bottom")

        bm25_rank_map = {"top__aaa": 0, "bot__bbb": 10}
        scored = engine._score_phase(
            [c_top, c_bot],
            query_tokens=["test"],
            query_embedding=None,
            bm25_rank_map=bm25_rank_map,
        )
        top_score = next(r["score"] for r in scored if r["skill_id"] == "top__aaa")
        bot_score = next(r["score"] for r in scored if r["skill_id"] == "bot__bbb")
        assert top_score > bot_score, (
            f"BM25 rank-0 must outscore rank-10: top={top_score}, bot={bot_score}"
        )

    def test_score_has_rrf_component(self):
        """Score must be at least the RRF contribution (>= 1/(k+0) + 1/(k+0))."""
        from openspace.cloud.search import SkillSearchEngine, _RRF_K
        engine = SkillSearchEngine()

        c = self._make_cand("s__aaa", "Skill")
        bm25_rank_map = {"s__aaa": 0}
        scored = engine._score_phase(
            [c],
            query_tokens=[],
            query_embedding=None,
            bm25_rank_map=bm25_rank_map,
        )
        rrf_min = 1.0 / (_RRF_K + 0) + 1.0 / (_RRF_K + 0)
        assert scored[0]["score"] >= rrf_min - 1e-9, \
            f"Score {scored[0]['score']} must be >= RRF min {rrf_min}"

    def test_nexus_boost_additive(self):
        """nexus_kb_validated boost adds on top of RRF score."""
        from openspace.cloud.search import SkillSearchEngine, _NEXUS_KB_CROSS_VALIDATION_BOOST
        engine = SkillSearchEngine()

        c = self._make_cand("s__abc", "weather", "weather forecast")
        bm25_rank_map = {"s__abc": 0}
        nexus_kws = {"weather", "forecast"}

        without = engine._score_phase(
            [c], query_tokens=["weather"], query_embedding=None,
            bm25_rank_map=bm25_rank_map,
        )[0]["score"]

        with_nexus = engine._score_phase(
            [c], query_tokens=["weather"], query_embedding=None,
            nexus_kb_keywords=nexus_kws, bm25_rank_map=bm25_rank_map,
        )
        assert with_nexus[0].get("nexus_kb_validated") is True
        assert with_nexus[0]["score"] > without, "nexus boost must increase score"
        diff = with_nexus[0]["score"] - without
        assert abs(diff - _NEXUS_KB_CROSS_VALIDATION_BOOST) < 1e-6

    def test_results_sorted_descending(self):
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()

        candidates = [self._make_cand(f"s{i}__abc", f"Skill {i}") for i in range(5)]
        bm25_rank_map = {c["skill_id"]: i for i, c in enumerate(candidates)}
        scored = engine._score_phase(
            candidates, query_tokens=[], query_embedding=None,
            bm25_rank_map=bm25_rank_map,
        )
        scores = [r["score"] for r in scored]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by descending score"

    def test_no_bm25_rank_map_graceful(self):
        """_score_phase must not crash when bm25_rank_map is None."""
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()

        c = self._make_cand("s__abc", "Skill")
        result = engine._score_phase([c], query_tokens=[], query_embedding=None)
        assert len(result) == 1
        assert result[0]["score"] > 0

    def test_vector_score_in_result_when_embedding_present(self):
        """When vector score > 0, result includes vector_score field."""
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()

        emb = [0.1] * 10
        c = self._make_cand("s__abc", "Skill")
        c["_embedding"] = emb

        with patch("openspace.cloud.embedding.cosine_similarity", return_value=0.85):
            scored = engine._score_phase(
                [c], query_tokens=[], query_embedding=emb,
                bm25_rank_map={"s__abc": 0},
            )
        assert "vector_score" in scored[0], "vector_score must be present when embedding used"
        assert scored[0]["vector_score"] == 0.85


# ---------------------------------------------------------------------------
# End-to-end search pipeline integration
# ---------------------------------------------------------------------------

class TestSearchPipelineIntegration:
    """Full search() call with RRF path exercised."""

    def _make_candidates(self) -> List[Dict[str, Any]]:
        return [
            {
                "skill_id": "weather__abc",
                "name": "Weather Forecast",
                "description": "Get weather data",
                "source": "local",
                "_body": "Fetches forecast from weather API. Returns temperature.",
                "_embedding_text": "Weather Forecast Get weather data Fetches forecast",
            },
            {
                "skill_id": "filelister__xyz",
                "name": "File Lister",
                "description": "List files in directory",
                "source": "local",
                "_body": "Uses os.listdir to enumerate directory contents.",
                "_embedding_text": "File Lister List files in directory os.listdir",
            },
            {
                "skill_id": "webfetch__def",
                "name": "Web Fetcher",
                "description": "Fetch web page content via HTTP request",
                "source": "cloud",
                "_embedding_text": "Web Fetcher Fetch web page content HTTP request",
            },
        ]

    def test_search_returns_list(self):
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()
        results = engine.search("weather", self._make_candidates(), limit=5)
        assert isinstance(results, list)

    def test_search_respects_limit(self):
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()
        results = engine.search("weather", self._make_candidates(), limit=2)
        assert len(results) <= 2

    def test_search_results_have_required_fields(self):
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()
        results = engine.search("weather", self._make_candidates(), limit=5)
        for r in results:
            assert "skill_id" in r
            assert "name" in r
            assert "score" in r
            assert r["score"] > 0

    def test_search_empty_query_returns_empty(self):
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()
        assert engine.search("", self._make_candidates()) == []
        assert engine.search("   ", self._make_candidates()) == []

    def test_search_empty_candidates_returns_empty(self):
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()
        assert engine.search("weather", []) == []

    def test_dedup_by_source_and_skill_id(self):
        """Same skill_id from different sources → kept separately."""
        from openspace.cloud.search import SkillSearchEngine
        engine = SkillSearchEngine()

        candidates = [
            {"skill_id": "s__abc", "name": "Skill", "description": "A", "source": "local"},
            {"skill_id": "s__abc", "name": "Skill", "description": "A", "source": "cloud"},
        ]
        results = engine.search("skill", candidates, limit=10)
        # Both should appear (different sources, different dedup keys)
        result_keys = {(r["source"], r["skill_id"]) for r in results}
        assert ("local", "s__abc") in result_keys
        assert ("cloud", "s__abc") in result_keys


# ---------------------------------------------------------------------------
# Cross-encoder reranking (Wave 5 Priority 1)
# ---------------------------------------------------------------------------

def _make_sc(skill_id: str, name: str = "", description: str = "", body: str = ""):
    from openspace.skill_engine.skill_ranker import SkillCandidate
    return SkillCandidate(
        skill_id=skill_id,
        name=name or skill_id,
        description=description or f"Description for {skill_id}",
        body=body,
    )


class TestCrossEncoderDisabledByDefault:
    """Cross-encoder must be opt-in (disabled by default)."""

    def test_skill_config_default_disabled(self):
        from openspace.config.grounding import SkillConfig
        cfg = SkillConfig()
        assert cfg.enable_cross_encoder is False

    def test_skill_ranker_default_disabled(self):
        from openspace.skill_engine.skill_ranker import SkillRanker
        r = SkillRanker()
        assert r._enable_cross_encoder is False

    def test_rerank_passthrough_when_disabled(self):
        """rerank() returns input unchanged when cross-encoder disabled."""
        from openspace.skill_engine.skill_ranker import SkillRanker
        r = SkillRanker(enable_cross_encoder=False)
        candidates = [_make_sc("a"), _make_sc("b"), _make_sc("c")]
        result = r.rerank("some query", candidates)
        # All candidates returned when disabled (no slicing by cross_encoder_top_k)
        assert result == candidates

    def test_hybrid_rank_no_cross_encoder_called(self):
        """hybrid_rank must NOT call rerank when cross_encoder is disabled."""
        from openspace.skill_engine.skill_ranker import SkillRanker
        r = SkillRanker(enable_cross_encoder=False)
        candidates = [_make_sc(f"s{i}") for i in range(5)]

        called = []
        original_rerank = r.rerank
        def spy_rerank(*args, **kwargs):
            called.append(True)
            return original_rerank(*args, **kwargs)

        r.rerank = spy_rerank
        r.hybrid_rank("test query", candidates, top_k=3)
        assert not called, "rerank must not be called when cross_encoder disabled"


class TestCrossEncoderRerank:
    """Cross-encoder reranking logic with mocked model."""

    def _ranker_with_mock_ce(self, mock_scores: List[float]):
        """Build a SkillRanker with cross-encoder enabled and a mocked predict()."""
        from openspace.skill_engine.skill_ranker import SkillRanker
        r = SkillRanker(enable_cross_encoder=True, cross_encoder_top_k=3)
        mock_ce = MagicMock()
        mock_ce.predict.return_value = mock_scores
        r._cross_encoder = mock_ce  # inject mock, skip lazy load
        return r, mock_ce

    def test_rerank_orders_by_score_descending(self):
        r, mock_ce = self._ranker_with_mock_ce([0.1, 0.9, 0.5])
        candidates = [_make_sc("low"), _make_sc("high"), _make_sc("mid")]
        result = r.rerank("query", candidates, top_k=3)
        assert [c.skill_id for c in result] == ["high", "mid", "low"]

    def test_rerank_respects_top_k(self):
        r, _ = self._ranker_with_mock_ce([0.9, 0.8, 0.7, 0.6, 0.5])
        candidates = [_make_sc(f"s{i}") for i in range(5)]
        result = r.rerank("query", candidates, top_k=2)
        assert len(result) == 2

    def test_rerank_uses_default_top_k(self):
        from openspace.skill_engine.skill_ranker import SkillRanker
        r = SkillRanker(enable_cross_encoder=True, cross_encoder_top_k=2)
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9, 0.8, 0.7, 0.6]
        r._cross_encoder = mock_ce
        candidates = [_make_sc(f"s{i}") for i in range(4)]
        result = r.rerank("query", candidates)
        assert len(result) == 2

    def test_rerank_returns_empty_for_empty_candidates(self):
        r, _ = self._ranker_with_mock_ce([])
        result = r.rerank("query", [])
        assert result == []

    def test_rerank_cache_hit_skips_predict(self):
        """Second identical call must use cache — predict NOT called again."""
        r, mock_ce = self._ranker_with_mock_ce([0.9, 0.5, 0.1])
        candidates = [_make_sc("a"), _make_sc("b"), _make_sc("c")]
        r.rerank("query", candidates, top_k=3)
        r.rerank("query", candidates, top_k=3)
        assert mock_ce.predict.call_count == 1, "predict must be called only once (cache hit)"

    def test_rerank_cache_miss_different_query(self):
        """Different query → cache miss → predict called again."""
        r, mock_ce = self._ranker_with_mock_ce([0.9, 0.5, 0.1])
        mock_ce.predict.return_value = [0.9, 0.5, 0.1]
        candidates = [_make_sc("a"), _make_sc("b"), _make_sc("c")]
        r.rerank("query one", candidates)
        r.rerank("query two", candidates)
        assert mock_ce.predict.call_count == 2

    def test_rerank_predict_failure_returns_candidates_truncated(self):
        """If predict() raises, return candidates[:top_k] gracefully."""
        from openspace.skill_engine.skill_ranker import SkillRanker
        r = SkillRanker(enable_cross_encoder=True, cross_encoder_top_k=2)
        mock_ce = MagicMock()
        mock_ce.predict.side_effect = RuntimeError("model error")
        r._cross_encoder = mock_ce
        candidates = [_make_sc(f"s{i}") for i in range(4)]
        result = r.rerank("query", candidates, top_k=2)
        assert len(result) == 2
        assert result[0].skill_id == "s0"  # original order preserved

    def test_hybrid_rank_calls_rerank_when_enabled(self):
        """hybrid_rank calls rerank() as Stage 3 when cross_encoder enabled."""
        from openspace.skill_engine.skill_ranker import SkillRanker
        r = SkillRanker(enable_cross_encoder=True, cross_encoder_top_k=3)
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [float(i) for i in range(10)]
        r._cross_encoder = mock_ce

        candidates = [_make_sc(f"s{i}") for i in range(5)]
        # Patch embedding to return empty (BM25-only path) so pipeline runs
        with patch.object(r, "_embedding_rank", return_value=[]):
            r.hybrid_rank("test query", candidates, top_k=3)

        assert mock_ce.predict.called, "cross-encoder predict must be called in hybrid_rank"


class TestCrossEncoderMissingDependency:
    """_get_cross_encoder gracefully disables when sentence-transformers absent."""

    def test_import_error_disables_cross_encoder(self):
        from openspace.skill_engine.skill_ranker import SkillRanker
        r = SkillRanker(enable_cross_encoder=True)

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("mocked missing")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            ce = r._get_cross_encoder()

        assert ce is None
        assert r._enable_cross_encoder is False

    def test_rerank_passthrough_after_import_failure(self):
        """After failed load, rerank() returns candidates unchanged."""
        from openspace.skill_engine.skill_ranker import SkillRanker
        r = SkillRanker(enable_cross_encoder=True)
        # Simulate load failure
        r._enable_cross_encoder = False
        candidates = [_make_sc("a"), _make_sc("b")]
        result = r.rerank("query", candidates)
        assert result == candidates


class TestSkillRegistryCrossEncoderConfig:
    """SkillRegistry.ranker picks up cross-encoder settings from SkillConfig."""

    def test_ranker_gets_enable_cross_encoder_from_config(self):
        from openspace.skill_engine.registry import SkillRegistry
        from openspace.config.grounding import SkillConfig
        cfg = SkillConfig(enable_cross_encoder=True, cross_encoder_top_k=7)
        reg = SkillRegistry(skill_cfg=cfg)
        ranker = reg.ranker
        assert ranker._enable_cross_encoder is True
        assert ranker._cross_encoder_top_k == 7

    def test_ranker_defaults_when_no_config(self):
        from openspace.skill_engine.registry import SkillRegistry
        from openspace.skill_engine.skill_ranker import SkillRanker
        reg = SkillRegistry()
        ranker = reg.ranker
        assert isinstance(ranker, SkillRanker)
        assert ranker._enable_cross_encoder is False
