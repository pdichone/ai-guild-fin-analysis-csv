import os
import json
import hashlib
from typing import Any, Dict, Optional
from datetime import datetime, timedelta


class CacheManager:
    def __init__(self, cache_dir: str = "cache", expiration_hours: int = 24):
        self.cache_dir = cache_dir
        self.expiration_delta = timedelta(hours=expiration_hours)
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, key: str) -> str:
        """Generate a unique cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def set(self, key: str, value: Any) -> None:
        """Store data in cache with timestamp."""
        cache_key = self._get_cache_key(key)
        cache_data = {"timestamp": datetime.now().isoformat(), "data": value}

        with open(self._get_cache_path(cache_key), "w") as f:
            json.dump(cache_data, f)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from cache if not expired."""
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            # Check expiration
            cached_time = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - cached_time > self.expiration_delta:
                os.remove(cache_path)
                return None

            return cache_data["data"]
        except Exception:
            return None

    def clear(self) -> None:
        """Clear all cached data."""
        for file in os.listdir(self.cache_dir):
            if file.endswith(".json"):
                os.remove(os.path.join(self.cache_dir, file))

    def clear_expired(self) -> None:
        """Clear only expired cache entries."""
        for file in os.listdir(self.cache_dir):
            if not file.endswith(".json"):
                continue

            try:
                with open(os.path.join(self.cache_dir, file), "r") as f:
                    cache_data = json.load(f)

                cached_time = datetime.fromisoformat(cache_data["timestamp"])
                if datetime.now() - cached_time > self.expiration_delta:
                    os.remove(os.path.join(self.cache_dir, file))
            except Exception:
                # Remove corrupted cache files
                os.remove(os.path.join(self.cache_dir, file))
