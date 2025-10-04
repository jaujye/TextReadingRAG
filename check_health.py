import requests
import json

try:
    r = requests.get('http://192.168.0.118:8080/health/detailed', timeout=5)
    data = r.json()

    print("=== Cache Status ===")
    cache = data.get('components', {}).get('cache', {})
    print(json.dumps(cache, indent=2))

    if cache.get('enabled'):
        print("\n✓ Cache is ENABLED")
    else:
        print("\n✗ Cache is DISABLED")

except Exception as e:
    print(f"Error: {e}")
