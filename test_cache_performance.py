#!/usr/bin/env python3
"""Test cache performance with queries."""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def get_cache_stats():
    """Get cache statistics."""
    try:
        r = requests.get(f"{BASE_URL}/health/detailed")
        data = r.json()
        cache = data.get("components", {}).get("cache", {})
        return cache
    except:
        return None

def test_query(query_text, collection="chinese_test"):
    """Test a query and measure time."""
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/api/query/",
        json={
            "query": query_text,
            "collection_name": collection,
            "retrieval_strategy": "hybrid",
            "top_k": 5
        },
        timeout=30
    )
    elapsed = (time.time() - start) * 1000

    if response.status_code == 200:
        return elapsed, True
    else:
        print(f"Query failed: {response.status_code} - {response.text[:200]}")
        return elapsed, False

print("=== Cache Performance Test ===\n")

# Get initial cache stats
stats = get_cache_stats()
if stats:
    print(f"Initial cache stats:")
    print(f"  Hits: {stats.get('keyspace_hits', 0)}")
    print(f"  Misses: {stats.get('keyspace_misses', 0)}")
    print(f"  Memory: {stats.get('used_memory_human', 'N/A')}\n")

query = "這個系統有什麼技術特點？"

# First query (should be cache miss)
print(f"Query 1 (cache miss expected): '{query}'")
time1, success1 = test_query(query)
if success1:
    print(f"✓ Time: {time1:.0f}ms\n")

    time.sleep(1)

    # Second identical query (should be cache hit)
    print(f"Query 2 (cache hit expected): '{query}'")
    time2, success2 = test_query(query)
    if success2:
        print(f"✓ Time: {time2:.0f}ms\n")

        improvement = ((time1 - time2) / time1) * 100
        print(f"Performance improvement: {improvement:.1f}%")
        print(f"Time saved: {time1 - time2:.0f}ms\n")

        # Get final cache stats
        stats = get_cache_stats()
        if stats:
            print(f"Final cache stats:")
            print(f"  Hits: {stats.get('keyspace_hits', 0)}")
            print(f"  Misses: {stats.get('keyspace_misses', 0)}")
            print(f"  Memory: {stats.get('used_memory_human', 'N/A')}")

            total = stats.get('keyspace_hits', 0) + stats.get('keyspace_misses', 0)
            if total > 0:
                hit_rate = stats.get('keyspace_hits', 0) / total * 100
                print(f"  Hit rate: {hit_rate:.1f}%")
