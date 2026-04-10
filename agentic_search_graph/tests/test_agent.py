"""Unit tests for pure helper functions in src/langgraph_agent.py."""
from datetime import datetime, timedelta

from src.langgraph_agent import (
    _enrich_query_for_freshness,
    _filter_stale_results,
    _is_sports_query,
    _mark_unfiltered,
    _normalize_for_match,
    _wants_fresh_results,
)


# ==========================================
# _normalize_for_match
# ==========================================

class TestNormalizeForMatch:
    def test_lowercases(self):
        assert _normalize_for_match("HOLA") == "hola"

    def test_removes_accents(self):
        assert _normalize_for_match("áéíóú") == "aeiou"

    def test_combined(self):
        assert _normalize_for_match("Ayer") == "ayer"
        # ñ is not in the replacement list — only accented vowels are normalized.
        assert _normalize_for_match("España") == "españa"
        assert _normalize_for_match("Último") == "ultimo"

    def test_no_change_needed(self):
        assert _normalize_for_match("hello world") == "hello world"


# ==========================================
# _wants_fresh_results
# ==========================================

class TestWantsFreshResults:
    def test_english_recency_words(self):
        assert _wants_fresh_results("What's the weather today?")
        assert _wants_fresh_results("Yankees score yesterday")
        assert _wants_fresh_results("game last night")
        assert _wants_fresh_results("what is happening right now")
        assert _wants_fresh_results("latest news")
        assert _wants_fresh_results("the final score was")

    def test_spanish_recency_words(self):
        assert _wants_fresh_results("¿Qué pasó ayer?")
        assert _wants_fresh_results("partido de hoy")
        assert _wants_fresh_results("resultado en vivo")
        assert _wants_fresh_results("noticias de última hora")

    def test_non_recency_queries(self):
        assert not _wants_fresh_results("What is a neural network?")
        assert not _wants_fresh_results("How does TCP/IP work?")
        assert not _wants_fresh_results("Capital of France")
        assert not _wants_fresh_results("Explícame qué es LangGraph")


# ==========================================
# _is_sports_query
# ==========================================

class TestIsSportsQuery:
    def test_league_names(self):
        assert _is_sports_query("NBA finals tonight")
        assert _is_sports_query("NFL week 3 results")
        assert _is_sports_query("MLB standings")

    def test_team_names(self):
        assert _is_sports_query("Yankees win yesterday")
        assert _is_sports_query("Real Madrid match")
        assert _is_sports_query("Barcelona vs Atletico")

    def test_spanish_sports_terms(self):
        assert _is_sports_query("resultado del partido")
        assert _is_sports_query("marcador de hoy")

    def test_non_sports(self):
        assert not _is_sports_query("Python programming tutorial")
        assert not _is_sports_query("Stock market news")
        assert not _is_sports_query("Latest iPhone model")


# ==========================================
# _enrich_query_for_freshness
# ==========================================

class TestEnrichQueryForFreshness:
    def test_adds_today_date_when_no_date(self):
        today = datetime.now().date().strftime("%Y-%m-%d")
        result = _enrich_query_for_freshness("latest AI news")
        assert today in result

    def test_adds_yesterday_date(self):
        yesterday = (datetime.now() - timedelta(days=1)).date().strftime("%Y-%m-%d")
        result = _enrich_query_for_freshness("Yankees game yesterday")
        assert yesterday in result

    def test_does_not_add_date_if_already_present(self):
        result = _enrich_query_for_freshness("Yankees 2026-04-08 game")
        # Should appear exactly once
        assert result.count("2026-04-08") == 1

    def test_adds_final_score_for_sports(self):
        result = _enrich_query_for_freshness("Yankees game yesterday")
        assert "final score" in result

    def test_does_not_duplicate_final_score(self):
        result = _enrich_query_for_freshness("Yankees final score yesterday")
        assert result.count("final score") == 1

    def test_empty_query_returns_empty(self):
        assert _enrich_query_for_freshness("") == ""

    def test_spanish_yesterday(self):
        yesterday = (datetime.now() - timedelta(days=1)).date().strftime("%Y-%m-%d")
        result = _enrich_query_for_freshness("partido de ayer")
        assert yesterday in result


# ==========================================
# _filter_stale_results
# ==========================================

class TestFilterStaleResults:
    def _make_result(self, title: str, date: str) -> dict:
        return {"title": title, "url": f"http://example.com/{title}", "published_date": date}

    def test_keeps_fresh_results(self):
        raw = {"results": [self._make_result("Fresh", "2026-04-08")]}
        out = _filter_stale_results(raw, "2026-04-02")
        assert len(out["results"]) == 1
        assert out["results"][0]["title"] == "Fresh"

    def test_removes_stale_results(self):
        raw = {
            "results": [
                self._make_result("Fresh", "2026-04-08"),
                self._make_result("Stale", "2024-04-10"),
            ]
        }
        out = _filter_stale_results(raw, "2026-04-02")
        assert len(out["results"]) == 1
        assert out["results"][0]["title"] == "Fresh"

    def test_keeps_results_without_date(self):
        raw = {"results": [{"title": "No date", "url": "http://example.com", "published_date": ""}]}
        out = _filter_stale_results(raw, "2026-04-02")
        assert len(out["results"]) == 1

    def test_returns_original_when_all_stale(self):
        """If every result is stale, the original dict is returned unchanged so the
        LLM still receives something and can apply its own date validation."""
        raw = {
            "results": [
                self._make_result("Old A", "2024-01-01"),
                self._make_result("Old B", "2023-06-15"),
            ]
        }
        out = _filter_stale_results(raw, "2026-04-02")
        # Must return the two original results, not an empty list.
        assert len(out["results"]) == 2

    def test_non_dict_input_passthrough(self):
        assert _filter_stale_results("not a dict", "2026-04-02") == "not a dict"

    def test_empty_results_list(self):
        raw = {"results": []}
        out = _filter_stale_results(raw, "2026-04-02")
        assert out["results"] == []


# ==========================================
# _mark_unfiltered
# ==========================================

class TestMarkUnfiltered:
    def test_adds_warning_key(self):
        raw = {"results": []}
        out = _mark_unfiltered(raw)
        assert "_warning" in out
        assert "Date filter" in out["_warning"]

    def test_does_not_mutate_original(self):
        raw = {"results": []}
        _mark_unfiltered(raw)
        assert "_warning" not in raw

    def test_non_dict_passthrough(self):
        assert _mark_unfiltered("plain string") == "plain string"
        assert _mark_unfiltered(42) == 42
