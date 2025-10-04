import requests
r = requests.get('http://localhost:8000/health/detailed')
print(r.text)
