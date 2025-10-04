import requests
r = requests.get('http://localhost:8000/health/detailed')
print(r.status_code)
if r.status_code == 200:
    import json
    data = r.json()
    cache = data.get('components', {}).get('cache', {})
    print(json.dumps(cache, indent=2, ensure_ascii=False))
