"""Debug: test Gemini Vision call directly."""
import os, json
from PIL import Image
import numpy as np

# Load API key from .env
with open('.env') as f:
    for line in f:
        if line.startswith('GEMINI_API_KEY='):
            os.environ['GEMINI_API_KEY'] = line.strip().split('=', 1)[1]

api_key = os.environ['GEMINI_API_KEY']
print(f"API key loaded: {api_key[:10]}...")

# Create a brownish-green test image (potato-like)
arr = np.zeros((200, 200, 3), dtype=np.uint8)
arr[:, :, 0] = 140
arr[:, :, 1] = 110
arr[:, :, 2] = 70
img = Image.fromarray(arr, 'RGB')
path = 'static/uploads/_dbg_potato.jpg'
img.save(path)
print(f"Test image saved: {path}")

from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)

with open(path, 'rb') as f:
    img_bytes = f.read()

print(f"Image bytes size: {len(img_bytes)}")
print("Testing gemini-2.5-flash with inline image bytes...")

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            types.Part.from_text("What plant is in this image? Answer in one word.")
        ]
    )
    print("SUCCESS! Response:", response.text[:300])
except Exception as e:
    print(f"ERROR ({type(e).__name__}): {str(e)[:400]}")

os.remove(path)
print("Done.")
