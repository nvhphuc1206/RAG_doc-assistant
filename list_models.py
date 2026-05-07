"""List all Gemini models available to your API key.

Run: python list_models.py
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GOOGLE_API_KEY not set in .env")

genai.configure(api_key=api_key)

print(f"\n{'Model name':<50} {'Methods supported'}")
print("=" * 90)
for m in genai.list_models():
    methods = ", ".join(m.supported_generation_methods)
    print(f"{m.name:<50} {methods}")
