"""Redis cache service for TextReadingRAG."""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from src.core.config import Settings

logger = logging.getLogger(__name__)


class CacheService:
    """
    Cache service for storing and retrieving frequently accessed data.

    Provides caching for:
    - Query embeddings
    - Query expansion results
    - Retrieval results
    - Reranking results
    """

    # Cache key version for invalidation
    CACHE_VERSION = "v1"

    def __init__(self, redis_client: Optional[Any], settings: Settings):
        """
        Initialize cache service.

        Args:
            redis_client: Redis async client (None if caching disabled)
            settings: Application settings
        """
        self.redis = redis_client
        self.settings = settings
        self.enabled = redis_client is not None

        if not self.enabled:
            logger.info("Cache service initialized in disabled mode")
        else:
            logger.info(f"Cache service initialized with TTL={settings.app.cache_ttl}s")

    def _generate_hash(self, *args: Any) -> str:
        """
        Generate consistent hash from arguments.

        Args:
            *args: Values to hash

        Returns:
            SHA256 hash string
        """
        content = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _build_key(self, prefix: str, *components: Any) -> str:
        """
        Build cache key with version and hash.

        Args:
            prefix: Cache key prefix (e.g., "emb", "exp", "ret")
            *components: Key components to hash

        Returns:
            Formatted cache key
        """
        hash_part = self._generate_hash(*components)
        return f"{prefix}:{self.CACHE_VERSION}:{hash_part}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/disabled
        """
        if not self.enabled:
            return None

        try:
            value = await self.redis.get(key)
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            else:
                logger.debug(f"Cache miss: {key}")
                return None
        except Exception as e:
            logger.warning(f"Cache get error for key '{key}': {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (uses default if None)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            ttl = ttl or self.settings.app.cache_ttl
            serialized = json.dumps(value, default=str)
            await self.redis.setex(key, ttl, serialized)
            logger.debug(f"Cache set: {key} (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key '{key}': {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            await self.redis.delete(key)
            logger.debug(f"Cache delete: {key}")
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key '{key}': {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Redis key pattern (e.g., "ret:v1:*")

        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0

        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += await self.redis.delete(*keys)
                if cursor == 0:
                    break

            logger.info(f"Cache pattern delete: {pattern} ({deleted} keys)")
            return deleted
        except Exception as e:
            logger.warning(f"Cache pattern delete error for '{pattern}': {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.enabled:
            return {"enabled": False}

        try:
            info = await self.redis.info("stats")
            memory = await self.redis.info("memory")

            return {
                "enabled": True,
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "used_memory_human": memory.get("used_memory_human", "0"),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}

    # Specialized cache methods for different data types

    async def get_embedding(
        self,
        query: str,
        model: str,
    ) -> Optional[List[float]]:
        """
        Get cached query embedding.

        Args:
            query: Query text
            model: Embedding model name

        Returns:
            Cached embedding vector or None
        """
        key = self._build_key("emb", model, query)
        return await self.get(key)

    async def set_embedding(
        self,
        query: str,
        model: str,
        embedding: List[float],
    ) -> bool:
        """
        Cache query embedding.

        Args:
            query: Query text
            model: Embedding model name
            embedding: Embedding vector

        Returns:
            True if successful
        """
        key = self._build_key("emb", model, query)
        return await self.set(key, embedding, ttl=3600)  # 1 hour

    async def get_query_expansion(
        self,
        query: str,
        methods: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached query expansion results.

        Args:
            query: Original query
            methods: Expansion methods used

        Returns:
            Cached expansion results or None
        """
        key = self._build_key("exp", query, sorted(methods))
        return await self.get(key)

    async def set_query_expansion(
        self,
        query: str,
        methods: List[str],
        results: Dict[str, Any],
    ) -> bool:
        """
        Cache query expansion results.

        Args:
            query: Original query
            methods: Expansion methods used
            results: Expansion results

        Returns:
            True if successful
        """
        key = self._build_key("exp", query, sorted(methods))
        return await self.set(key, results, ttl=1800)  # 30 minutes

    async def get_retrieval(
        self,
        query: str,
        mode: str,
        collection: str,
        top_k: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached retrieval results.

        Args:
            query: Query text
            mode: Retrieval mode
            collection: Collection name
            top_k: Number of results

        Returns:
            Cached retrieval results or None
        """
        key = self._build_key("ret", query, mode, collection, top_k)
        return await self.get(key)

    async def set_retrieval(
        self,
        query: str,
        mode: str,
        collection: str,
        top_k: int,
        results: List[Dict[str, Any]],
    ) -> bool:
        """
        Cache retrieval results.

        Args:
            query: Query text
            mode: Retrieval mode
            collection: Collection name
            top_k: Number of results
            results: Retrieval results

        Returns:
            True if successful
        """
        key = self._build_key("ret", query, mode, collection, top_k)
        return await self.set(key, results, ttl=1800)  # 30 minutes

    async def invalidate_collection(self, collection: str) -> int:
        """
        Invalidate all cache entries for a collection.

        Args:
            collection: Collection name

        Returns:
            Number of keys deleted
        """
        # Retrieval results are collection-specific
        pattern = f"ret:{self.CACHE_VERSION}:*"
        deleted = await self.delete_pattern(pattern)
        logger.info(f"Invalidated cache for collection '{collection}': {deleted} keys")
        return deleted
