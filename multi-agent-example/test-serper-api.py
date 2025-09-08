import requests, json
API_KEY = "<<YOUR_SERPER_API_KEY_HERE,get from https://serper.dev/>>"
url = "https://google.serper.dev/search"
payload = json.dumps({"q": "Microsoft"})
headers = {
    "X-API-KEY": API_KEY,
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=payload)
print(response.json())
