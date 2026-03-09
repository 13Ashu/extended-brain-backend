import requests
import json

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

headers = {
    "Content-Type": "application/json",
    "X-goog-api-key": "AIzaSyAu_y_ffx4NJmQi0wUpGAq8tGoASf58iug"
}

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Explain how AI works in a few words"
                }
            ]
        }
    ]
}

response = requests.post(url, headers=headers, json=payload)
data = response.json()

# print(json.dumps(data, indent=2))

text = data["candidates"][0]["content"]["parts"][0]["text"]
print(text)