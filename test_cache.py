#!/usr/bin/env python3
"""Test cache functionality."""

import requests
import json
import time

BASE_URL = "http://192.168.0.118:8080"

def test_health_check():
    """Test health check shows cache enabled."""
    print("\n=== Testing Health Check ===")

    response = requests.get(f"{BASE_URL}/health/detailed")
    data = response.json()

    print(json.dumps(data, indent=2))

    cache_status = data.get("components", {}).get("cache", {})
    print(f"\nCache Enabled: {cache_status.get('enabled')}")
    print(f"Cache Status: {cache_status.get('status')}")

    if cache_status.get('enabled'):
        print(f"Redis Host: {cache_status.get('redis_host')}")
        if 'keyspace_hits' in cache_status:
            print(f"Cache Hits: {cache_status.get('keyspace_hits')}")
            print(f"Cache Misses: {cache_status.get('keyspace_misses')}")
            total = cache_status.get('keyspace_hits', 0) + cache_status.get('keyspace_misses', 0)
            if total > 0:
                hit_rate = cache_status.get('keyspace_hits', 0) / total * 100
                print(f"Hit Rate: {hit_rate:.1f}%")

    return cache_status.get('enabled', False)

def test_query_performance():
    """Test query performance with cache."""
    print("\n=== Testing Query Performance ===")

    query = "這個系統有什麼技術特點？"
    collection = "chinese_test"

    # First query (cache miss)
    print(f"\nQuery 1 (cache miss expected): {query}")
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "query": query,
            "collection_name": collection,
            "retrieval_strategy": "hybrid",
            "top_k": 5
        }
    )
    first_time = (time.time() - start) * 1000

    if response.status_code == 200:
        print(f"✓ First query time: {first_time:.0f}ms")
    else:
        print(f"✗ First query failed: {response.status_code}")
        return

    # Wait a bit
    time.sleep(1)

    # Second identical query (cache hit expected)
    print(f"\nQuery 2 (cache hit expected): {query}")
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "query": query,
            "collection_name": collection,
            "retrieval_strategy": "hybrid",
            "top_k": 5
        }
    )
    second_time = (time.time() - start) * 1000

    if response.status_code == 200:
        print(f"✓ Second query time: {second_time:.0f}ms")
    else:
        print(f"✗ Second query failed: {response.status_code}")
        return

    # Calculate improvement
    if first_time > second_time:
        improvement = ((first_time - second_time) / first_time) * 100
        print(f"\n✓ Performance improvement: {improvement:.1f}% faster ({first_time-second_time:.0f}ms saved)")
    else:
        print(f"\n⚠ Second query was slower (possible cache miss)")

if __name__ == "__main__":
    print("=== Redis Cache Integration Test ===")

    cache_enabled = test_health_check()

    if cache_enabled:
        test_query_performance()

        # Check final cache stats
        print("\n=== Final Cache Stats ===")
        test_health_check()
    else:
        print("\n✗ Cache is not enabled!")
