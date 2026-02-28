import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from tavily import TavilyClient

logger = logging.getLogger(__name__)


class SearchCache:
    """Thread-safe search cache with in-memory store backed by disk persistence."""

    def __init__(self, max_retries: int = 3, initial_delay: float = 2.0) -> None:
        self.max_retries: int = max_retries
        self.initial_delay: float = initial_delay
        self.last_request_time: float = 0.0
        self.min_request_interval: float = 0.7
        self._lock = threading.Lock()
        self._cache: Dict[str, str] = {}
        self._loaded = False

    def get_cache_path(self, custom_path: Optional[Path] = None) -> Path:
        cache_dir = custom_path or Path("data/examples")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "search_cache.json"

    def _ensure_loaded(self, cache_path: Optional[Path] = None) -> None:
        """Lazy-load cache from disk on first access. Must be called under self._lock."""
        if self._loaded:
            return
        path = self.get_cache_path(cache_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Cache file corrupted, creating new cache")
                self._cache = {}
        self._loaded = True

    def get(self, key: str, cache_path: Optional[Path] = None) -> Optional[str]:
        """Return cached value for key, or None if not found."""
        with self._lock:
            self._ensure_loaded(cache_path)
            return self._cache.get(key)

    def put(self, key: str, value: str, cache_path: Optional[Path] = None) -> None:
        """Store a value and persist to disk."""
        with self._lock:
            self._ensure_loaded(cache_path)
            self._cache[key] = value
            self._save_to_disk(cache_path)

    def _save_to_disk(self, cache_path: Optional[Path] = None) -> None:
        """Write the in-memory cache to disk. Must be called under self._lock."""
        path = self.get_cache_path(cache_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def rate_limit(self) -> None:
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self.last_request_time = time.time()


def perform_online_search(
    expense_name: str,
    tavily_client: TavilyClient,
    search_cache: SearchCache,
    max_results: int = 2,
    cache_path: Optional[Path] = None,
) -> str:
    """
    Search for expense details using Tavily with caching and rate limiting.

    Args:
        expense_name: Raw expense name to search for.
        tavily_client: Tavily API client instance.
        search_cache: SearchCache instance for caching results.
        max_results: Maximum number of search results. Defaults to 2.
        cache_path: Custom path for cache file. Defaults to None.

    Returns:
        JSON-formatted search results string or error message.
    """
    logger.debug(f"Starting search for expense: '{expense_name}'")

    # Clean input
    texts_to_remove = ["SumUp  *", "PAYPAL *", "LSP*", "CRV*", "PAY.nl*", "UZR*", "luca "]
    cleaned_name = expense_name
    for text in texts_to_remove:
        cleaned_name = cleaned_name.replace(text, "")
    cleaned_name = cleaned_name.strip()

    if not cleaned_name:
        logger.warning(f"Invalid search term after cleaning: {expense_name}")
        return "Invalid search term"

    # Check cache
    cache_key = f"{cleaned_name}:{max_results}"
    cached = search_cache.get(cache_key, cache_path)
    if cached is not None:
        logger.debug(f"Returning cached results for: {cleaned_name}")
        return cached

    logger.debug(f"Performing online search as no cached values for: {cleaned_name}")

    # Implement exponential backoff
    for attempt in range(search_cache.max_retries):
        try:
            search_cache.rate_limit()
            logger.debug(f"Search attempt {attempt + 1} for: {cleaned_name}")

            search_results = tavily_client.search(
                query=cleaned_name,
                search_depth="basic",
                max_results=max_results,
                country="germany",
            )

            # Fallback: retry without country filter if no results
            if not search_results or not search_results.get("results"):
                logger.debug(f"No results with country filter, retrying without for: {cleaned_name}")
                search_cache.rate_limit()
                search_results = tavily_client.search(
                    query=cleaned_name,
                    search_depth="basic",
                    max_results=max_results,
                )

            search_result_str: str
            if search_results and search_results.get("results"):
                search_result_str = json.dumps(search_results["results"], ensure_ascii=False)
                logger.debug(f"Found {len(search_results['results'])} results for: {cleaned_name}")
                search_cache.put(cache_key, search_result_str, cache_path)
            else:
                search_result_str = "No results found"
                logger.warning(f"No results found for '{cleaned_name}'")

            return search_result_str

        except Exception as e:
            delay = search_cache.initial_delay * (2**attempt)
            logger.warning(f"Search attempt {attempt + 1} failed for '{cleaned_name}': {str(e)}. Retrying in {delay}s")
            time.sleep(delay)

    logger.error(f"Search failed after {search_cache.max_retries} attempts for: {cleaned_name}")
    return "Online search failed after multiple attempts"
