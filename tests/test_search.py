"""Tests for the search module."""

import json
import time
from unittest.mock import MagicMock, patch

from import_bank_details.search import SearchCache, perform_online_search


class TestSearchCache:
    """Tests for SearchCache class."""

    def test_init(self):
        """Test SearchCache initialization."""
        cache = SearchCache()
        assert cache.max_retries == 3
        assert cache.initial_delay == 2.0
        assert cache.last_request_time == 0.0
        assert cache.min_request_interval == 0.7
        assert cache._cache == {}
        assert cache._loaded is False

    def test_two_instances_are_independent(self):
        """Test that two SearchCache instances do not share state."""
        cache1 = SearchCache()
        cache2 = SearchCache()
        cache1._cache["key"] = "value"
        cache1._loaded = True
        assert cache2._cache == {}
        assert cache2._loaded is False

    def test_get_cache_path_default(self):
        """Test get_cache_path with default path."""
        cache = SearchCache()
        default_path = cache.get_cache_path()
        assert str(default_path).endswith("data/examples/search_cache.json")

    def test_get_cache_path_custom(self, tmp_path):
        """Test get_cache_path with custom path."""
        cache = SearchCache()
        custom_path = cache.get_cache_path(tmp_path)
        assert custom_path == tmp_path / "search_cache.json"
        assert tmp_path.exists()

    def test_get_returns_none_for_missing_key(self, tmp_path):
        """Test get() returns None for missing key."""
        cache = SearchCache()
        assert cache.get("missing_key", tmp_path) is None

    def test_put_and_get(self, tmp_path):
        """Test put() stores value and get() retrieves it."""
        cache = SearchCache()
        cache.put("key1", "value1", tmp_path)
        assert cache.get("key1", tmp_path) == "value1"

    def test_put_persists_to_disk(self, tmp_path):
        """Test put() persists values to disk."""
        cache = SearchCache()
        cache.put("key1", "value1", tmp_path)

        # Read from disk directly
        cache_file = tmp_path / "search_cache.json"
        assert cache_file.exists()
        with open(cache_file) as f:
            data = json.load(f)
        assert data == {"key1": "value1"}

    def test_lazy_load_from_disk(self, tmp_path):
        """Test that cache lazy-loads from disk on first access."""
        # Write data to disk first
        cache_file = tmp_path / "search_cache.json"
        with open(cache_file, "w") as f:
            json.dump({"existing_key": "existing_value"}, f)

        # Create new cache instance - should lazy-load
        cache = SearchCache()
        assert cache.get("existing_key", tmp_path) == "existing_value"

    def test_corrupted_cache_file(self, tmp_path):
        """Test that corrupted cache file is handled gracefully."""
        cache_file = tmp_path / "search_cache.json"
        cache_file.write_text("invalid json{{{")

        cache = SearchCache()
        assert cache.get("any_key", tmp_path) is None
        # Should still work after corruption
        cache.put("new_key", "new_value", tmp_path)
        assert cache.get("new_key", tmp_path) == "new_value"

    def test_rate_limit_first_call_fast(self):
        """Test that first rate_limit call does not sleep."""
        cache = SearchCache()
        cache.min_request_interval = 0.1

        start_time = time.time()
        cache.rate_limit()
        elapsed = time.time() - start_time
        assert elapsed < 0.05

    def test_rate_limit_second_call_sleeps(self):
        """Test that second immediate rate_limit call sleeps."""
        cache = SearchCache()
        cache.min_request_interval = 0.1
        cache.rate_limit()

        with patch("time.sleep") as mock_sleep:
            cache.rate_limit()
            mock_sleep.assert_called_once()


class TestPerformOnlineSearch:
    """Tests for perform_online_search function."""

    def test_basic_search(self):
        """Test basic search with country filter on first call."""
        mock_tavily = MagicMock()
        mock_results = {"results": [{"title": "Test Result", "content": "Test Content"}]}
        mock_tavily.search.return_value = mock_results

        cache = SearchCache()
        cache._loaded = True  # Skip disk loading

        result = perform_online_search("test query", mock_tavily, cache)

        mock_tavily.search.assert_called_once_with(query="test query", search_depth="basic", max_results=2, country="germany")
        assert "Test Result" in result
        assert "Test Content" in result

    def test_cached_result(self, tmp_path):
        """Test that cached results are returned without API call."""
        mock_tavily = MagicMock()

        cache = SearchCache()
        cache._loaded = True
        cache._cache["test query:2"] = "Cached result"

        result = perform_online_search("test query", mock_tavily, cache, cache_path=tmp_path)

        mock_tavily.search.assert_not_called()
        assert result == "Cached result"

    def test_empty_results_not_cached(self):
        """Test search with empty results is not cached."""
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {"results": []}

        cache = SearchCache()
        cache._loaded = True

        result = perform_online_search("test query", mock_tavily, cache)
        assert result == "No results found"
        # "No results found" must NOT be cached
        assert cache.get("test query:2") is None

    def test_fallback_without_country(self):
        """Test fallback search without country filter when first call returns empty."""
        mock_tavily = MagicMock()
        empty_results: dict[str, list[dict[str, str]]] = {"results": []}
        success_results = {"results": [{"title": "Found It", "content": "International"}]}
        mock_tavily.search.side_effect = [empty_results, success_results]

        cache = SearchCache()
        cache._loaded = True

        result = perform_online_search("RICEPAELLA", mock_tavily, cache)

        assert mock_tavily.search.call_count == 2
        # First call with country filter
        mock_tavily.search.assert_any_call(query="RICEPAELLA", search_depth="basic", max_results=2, country="germany")
        # Second call without country filter
        mock_tavily.search.assert_any_call(query="RICEPAELLA", search_depth="basic", max_results=2)
        assert "Found It" in result
        # Successful result should be cached
        assert cache.get("RICEPAELLA:2") is not None

    def test_fallback_both_empty(self):
        """Test that both calls returning empty gives 'No results found' and is NOT cached."""
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {"results": []}

        cache = SearchCache()
        cache._loaded = True

        result = perform_online_search("Contipark 09033051601", mock_tavily, cache)

        assert mock_tavily.search.call_count == 2
        assert result == "No results found"
        # Must NOT be cached
        assert cache.get("Contipark 09033051601:2") is None

    def test_invalid_search_term(self):
        """Test search with term that becomes empty after cleaning."""
        mock_tavily = MagicMock()
        cache = SearchCache()
        cache._loaded = True

        result = perform_online_search("SumUp  *", mock_tavily, cache)
        assert result == "Invalid search term"
        mock_tavily.search.assert_not_called()

    def test_cleaning_prefixes(self):
        """Test that common prefixes are cleaned from search terms."""
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {"results": [{"title": "Result"}]}

        cache = SearchCache()
        cache._loaded = True

        perform_online_search("PAYPAL *Coffee Shop", mock_tavily, cache)
        mock_tavily.search.assert_called_once_with(query="Coffee Shop", search_depth="basic", max_results=2, country="germany")

    @patch("time.sleep", return_value=None)
    def test_retry_and_fail(self, mock_sleep):
        """Test retry logic when all attempts fail."""
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = Exception("API limit exceeded")

        cache = SearchCache(max_retries=3, initial_delay=0.1)
        cache._loaded = True
        cache.min_request_interval = 0  # Disable rate limiting for this test

        result = perform_online_search("test query", mock_tavily, cache)

        assert mock_tavily.search.call_count == 3
        # 3 sleeps from retry backoff (rate_limit won't sleep with interval=0)
        assert mock_sleep.call_count == 3
        assert result == "Online search failed after multiple attempts"

    @patch("time.sleep", return_value=None)
    def test_retry_and_succeed(self, mock_sleep):
        """Test retry logic that eventually succeeds."""
        mock_tavily = MagicMock()
        mock_results = {"results": [{"title": "Test Result", "content": "Test Content"}]}
        mock_tavily.search.side_effect = [
            Exception("API limit exceeded"),
            Exception("Search error"),
            mock_results,
        ]

        cache = SearchCache(max_retries=3, initial_delay=0.1)
        cache._loaded = True
        cache.min_request_interval = 0  # Disable rate limiting for this test

        result = perform_online_search("test query", mock_tavily, cache)

        assert mock_tavily.search.call_count == 3
        # 2 sleeps from retry backoff for the 2 failed attempts
        assert mock_sleep.call_count == 2
        assert "Test Result" in result
        assert "Test Content" in result
