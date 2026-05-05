"""Check Gemini REST API quota - try different endpoints."""
import os, requests, json, io, base64
from PIL import Image
import numpy as np

with open('.env') as f:
    for line in f:
        if line.startswith('GEMINI_API_KEY='):
            os.environ['GEMINI_API_KEY'] = line.strip().split('=', 1)[1]

api_key = os.environ['GEMINI_API_KEY']

arr = np.full((32, 32, 3), [60, 150, 50], dtype=np.uint8)
buf = io.BytesIO()
Image.fromarray(arr, 'RGB').save(buf, format='JPEG')
b64 = base64.b64encode(buf.getvalue()).decode()

# Try both v1beta and v1
endpoints = [
    ("v1beta", "gemini-2.0-flash"),
    ("v1beta", "gemini-2.0-flash-lite"),
    ("v1",     "gemini-1.5-flash"),
    ("v1",     "gemini-1.5-flash-8b"),
    ("v1beta", "gemini-2.5-flash-preview-04-17"),
    ("v1beta", "gemini-2.5-pro-preview-03-25"),
]

for version, model in endpoints:
    url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": "One word answer: what color?"}
        ]}],
        "generationConfig": {"maxOutputTokens": 20}
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            print(f"  OK  [{version}] {model}: {text.strip()}")
        elif resp.status_code == 429:
            err = resp.json().get('error',{}).get('message','')[:80]
            print(f"  429 [{version}] {model}: QUOTA — {err}")
        else:
            print(f"  {resp.status_code} [{version}] {model}: {resp.text[:80]}")
    except Exception as e:
        print(f"  ERR [{version}] {model}: {e}")
