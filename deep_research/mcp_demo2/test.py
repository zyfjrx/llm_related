import httpx
import requests
import json

url = f'http://localhost:8080/search?q=小米汽车&format=json'

response = requests.get(url, timeout=10)
data = response.json()['results']


articles = [
    {
        "title": item.get("title"),
        "desc": item.get("content"),
        "url": item.get("url")
    } for item in data[:5]
]


print(articles)
