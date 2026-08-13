"""Thread-safe, disk-persisted cache of expense classifications by merchant name."""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

from import_bank_details.expense_names import clean_expense_name

logger = logging.getLogger(__name__)


class ClassificationCache:
    """Thread-safe classification cache with in-memory store backed by disk persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, str]] = {}
        self._loaded = False

    def get_cache_path(self, custom_path: Optional[Path] = None) -> Path:
        cache_dir = custom_path or Path("data/examples")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "classification_cache.json"

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

    def get(self, expense_name: str, cache_path: Optional[Path] = None) -> Optional[Dict[str, str]]:
        """Return cached classification for a cleaned merchant name, or None on miss."""
        cleaned_name = clean_expense_name(expense_name)
        if not cleaned_name:
            return None
        with self._lock:
            self._ensure_loaded(cache_path)
            return self._cache.get(cleaned_name)

    def put(
        self,
        expense_name: str,
        primary: str,
        secondary: str,
        cache_path: Optional[Path] = None,
    ) -> None:
        """Store a classification and persist to disk. Skips empty names or categories."""
        cleaned_name = clean_expense_name(expense_name)
        if not cleaned_name or not primary or not secondary:
            return
        with self._lock:
            self._ensure_loaded(cache_path)
            self._cache[cleaned_name] = {"Primary": primary, "Secondary": secondary}
            self._save_to_disk(cache_path)

    def _save_to_disk(self, cache_path: Optional[Path] = None) -> None:
        """Write the in-memory cache to disk. Must be called under self._lock."""
        path = self.get_cache_path(cache_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
