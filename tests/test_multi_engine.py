"""Tests for multi_engine.py — no real API calls."""

import pytest
from unittest.mock import MagicMock, patch

from printforge.multi_engine import MultiEngine, EngineResult, ComparisonResult


# ─────────────────────────────────────────────────────────────
# 1. MultiEngine initialization
# ─────────────────────────────────────────────────────────────

class TestMultiEngineInit:
    """Test MultiEngine.__init__."""

    def test_creates_instance(self):
        engine = MultiEngine()
        assert engine is not None

    def test_tripo_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TRIPO_API_KEY", "test-key-123")
        engine = MultiEngine()
        assert engine.tripo_key == "test-key-123"

    def test_tripo_key_defaults_to_empty(self, monkeypatch):
        monkeypatch.delenv("TRIPO_API_KEY", raising=False)
        engine = MultiEngine()
        assert engine.tripo_key == ""


# ─────────────────────────────────────────────────────────────
# 2. ENGINES config completeness
# ─────────────────────────────────────────────────────────────

class TestEnginesConfig:
    """Test that ENGINES dict is fully populated."""

    def test_engines_dict_not_empty(self):
        assert len(MultiEngine.ENGINES) > 0

    def test_engines_have_required_keys(self):
        required_keys = {"name", "model_version", "credits"}
        for engine_id, config in MultiEngine.ENGINES.items():
            assert required_keys.issubset(config.keys()), \
                f"{engine_id} missing keys: {required_keys - config.keys()}"

    def test_tripo_p1_config(self):
        cfg = MultiEngine.ENGINES["tripo_p1"]
        assert cfg["name"] == "Tripo P1 (Latest)"
        assert cfg["model_version"] == "P1-20260311"
        assert cfg["credits"] == 50

    def test_tripo_v3_config(self):
        cfg = MultiEngine.ENGINES["tripo_v3"]
        assert cfg["name"] == "Tripo v3.1"
        assert cfg["model_version"] == "v3.1-20260211"
        assert cfg["credits"] == 30

    def test_tripo_v2_config(self):
        cfg = MultiEngine.ENGINES["tripo_v2"]
        assert cfg["name"] == "Tripo v2.5 (Fast)"
        assert cfg["model_version"] == "v2.5-20250123"
        assert cfg["credits"] == 20


# ─────────────────────────────────────────────────────────────
# 3. _score_results scoring logic
# ─────────────────────────────────────────────────────────────

class TestScoreResults:
    """Test _score_results internal scoring heuristics."""

    def _make_comparison(self, results: dict) -> ComparisonResult:
        comp = ComparisonResult(results=results)
        # Inject the scorer manually (simulate what compare() does)
        MultiEngine()._score_results(comp)
        return comp

    def test_failed_result_scores_zero(self):
        failed = EngineResult(engine="tripo_p1", success=False, error="boom")
        comp = self._make_comparison({"tripo_p1": failed})
        assert comp.results["tripo_p1"].quality_score == 0.0

    def test_high_vertex_count_boosts_score(self):
        # >10k vertices: base 0.5 + 0.15 = 0.65
        result = EngineResult(
            engine="tripo_p1", success=True, vertices=15000,
            faces=8000, has_texture=False,
            generation_time_s=90, glb_data=b"x" * 100_000,
        )
        comp = self._make_comparison({"tripo_p1": result})
        assert comp.results["tripo_p1"].quality_score == 0.65

    def test_medium_vertex_count_boosts_score(self):
        # 5k-10k: base 0.5 + 0.1 = 0.6
        result = EngineResult(
            engine="tripo_p1", success=True, vertices=7000,
            faces=4000, has_texture=False,
            generation_time_s=90, glb_data=b"x" * 100_000,
        )
        comp = self._make_comparison({"tripo_p1": result})
        assert comp.results["tripo_p1"].quality_score == 0.6

    def test_texture_boosts_score(self):
        # has_texture=True: base 0.5 + 0.2 = 0.7
        result = EngineResult(
            engine="tripo_v3", success=True, vertices=2000,
            faces=1000, has_texture=True,
            generation_time_s=90, glb_data=b"x" * 100_000,
        )
        comp = self._make_comparison({"tripo_v3": result})
        assert comp.results["tripo_v3"].quality_score == 0.7

    def test_large_file_size_boosts_score(self):
        # >2MB: base 0.5 + 0.1 = 0.6
        result = EngineResult(
            engine="tripo_p1", success=True, vertices=2000,
            faces=1000, has_texture=False,
            generation_time_s=90, glb_data=b"x" * 2_500_000,
        )
        comp = self._make_comparison({"tripo_p1": result})
        assert comp.results["tripo_p1"].quality_score == 0.6

    def test_speed_bonus_applied(self):
        # <60s: base 0.5 + 0.05 = 0.55
        result = EngineResult(
            engine="tripo_v2", success=True, vertices=1000,
            faces=500, has_texture=False,
            generation_time_s=45, glb_data=b"x" * 100_000,
        )
        comp = self._make_comparison({"tripo_v2": result})
        assert comp.results["tripo_v2"].quality_score == 0.55

    def test_score_caps_at_one(self):
        # All bonuses stacked: 0.5 + 0.15 + 0.2 + 0.1 + 0.05 = 1.0
        result = EngineResult(
            engine="tripo_p1", success=True, vertices=15000,
            faces=8000, has_texture=True,
            generation_time_s=30, glb_data=b"x" * 3_000_000,
        )
        comp = self._make_comparison({"tripo_p1": result})
        assert comp.results["tripo_p1"].quality_score == 1.0


# ─────────────────────────────────────────────────────────────
# 4. ComparisonResult.successful filtering
# ─────────────────────────────────────────────────────────────

class TestComparisonResultSuccessful:
    """Test the successful property filters correctly."""

    def test_empty_results_returns_empty_list(self):
        comp = ComparisonResult(results={})
        assert comp.successful == []

    def test_only_successful_returned(self):
        r1 = EngineResult(engine="tripo_p1", success=True)
        r2 = EngineResult(engine="tripo_v3", success=False, error="fail")
        r3 = EngineResult(engine="tripo_v2", success=True)
        comp = ComparisonResult(results={
            "tripo_p1": r1,
            "tripo_v3": r2,
            "tripo_v2": r3,
        })
        successful = comp.successful
        assert len(successful) == 2
        assert all(r.success for r in successful)
        assert r2 not in successful

    def test_all_failed_returns_empty(self):
        r1 = EngineResult(engine="tripo_p1", success=False)
        r2 = EngineResult(engine="tripo_v3", success=False)
        comp = ComparisonResult(results={"tripo_p1": r1, "tripo_v3": r2})
        assert comp.successful == []


# ─────────────────────────────────────────────────────────────
# 5. Recommendation logic — highest score wins
# ─────────────────────────────────────────────────────────────

class TestRecommendation:
    """Test _score_results sets recommendation correctly."""

    def _score_and_get(self, results: dict) -> ComparisonResult:
        comp = ComparisonResult(results=results)
        MultiEngine()._score_results(comp)
        return comp

    def test_recommends_highest_score(self):
        low = EngineResult(
            engine="tripo_v2", success=True, vertices=2000,
            faces=1000, has_texture=False,
            generation_time_s=90, glb_data=b"x" * 100_000,
        )
        high = EngineResult(
            engine="tripo_p1", success=True, vertices=15000,
            faces=8000, has_texture=True,
            generation_time_s=30, glb_data=b"x" * 3_000_000,
        )
        comp = self._score_and_get({"tripo_v2": low, "tripo_p1": high})
        assert comp.recommended == "tripo_p1"

    def test_no_recommendation_when_all_fail(self):
        r1 = EngineResult(engine="tripo_p1", success=False, error="boom")
        r2 = EngineResult(engine="tripo_v3", success=False, error="nope")
        comp = self._score_and_get({"tripo_p1": r1, "tripo_v3": r2})
        assert comp.recommended is None

    def test_empty_comparison_no_recommendation(self):
        comp = self._score_and_get({})
        assert comp.recommended is None

    def test_single_successful_is_recommended(self):
        r = EngineResult(
            engine="tripo_v3", success=True, vertices=5000,
            faces=2500, has_texture=True,
            generation_time_s=40, glb_data=b"x" * 600_000,
        )
        comp = self._score_and_get({"tripo_v3": r})
        assert comp.recommended == "tripo_v3"

    def test_tie_goes_to_first_max(self):
        # Same score — behaviour is undefined; just ensure one is chosen
        r1 = EngineResult(
            engine="tripo_p1", success=True, vertices=1000,
            faces=500, has_texture=False,
            generation_time_s=90, glb_data=b"x" * 100_000,
        )
        r2 = EngineResult(
            engine="tripo_v3", success=True, vertices=1000,
            faces=500, has_texture=False,
            generation_time_s=90, glb_data=b"x" * 100_000,
        )
        comp = self._score_and_get({"tripo_p1": r1, "tripo_v3": r2})
        assert comp.recommended in ("tripo_p1", "tripo_v3")
