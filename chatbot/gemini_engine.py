"""
=============================================================
  AgreeCulture - AI Engine
  Priority: Gemini (4 models) -> OpenRouter (3 free models) -> Local fallback
=============================================================
  FREE APIs used:
  1. Google Gemini - https://aistudio.google.com/
  2. OpenRouter    - https://openrouter.ai/  (free models available)
=============================================================
"""

import os
import json
import requests

# ─────────────────────────────────────────────────────────────────────────────
#  Load knowledge base
# ─────────────────────────────────────────────────────────────────────────────

def _load_knowledge_base() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_path  = os.path.join(base_dir, "data", "knowledge_base.json")
    try:
        with open(kb_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


_KB = _load_knowledge_base()


def _build_crop_summary() -> str:
    lines = []
    for key, crop in _KB.get("crops", {}).items():
        lines.append(
            f"- {crop['name']}: season={crop['season']}, soil={crop['soil_type']}, "
            f"water={crop['water_requirement']}, fertilizer={crop['fertilizer']}, "
            f"yield={crop['yield']}, diseases={', '.join(crop.get('common_diseases', []))}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  System prompt (shared by Gemini and OpenRouter)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are AgreeCulture Assistant, an expert AI farming advisor.

Your role is to help Indian farmers and agriculture students with:
- Crop growing guides and step-by-step cultivation advice
- Fertilizer recommendations (NPK, organic, micronutrients)
- Water and irrigation management
- Soil health and preparation
- Pest and disease identification and treatment
- Season and sowing timelines (Kharif, Rabi, Zaid)
- Expected yields and harvesting tips
- General agricultural best practices

KNOWLEDGE BASE (use this as your primary reference):
{_build_crop_summary()}

RESPONSE RULES:
1. Keep responses concise but helpful — 3 to 6 sentences or a short bullet list
2. Use simple language suitable for farmers and students
3. Use relevant emojis to make responses friendly
4. Format key terms in **bold** when helpful
5. For crops not in the knowledge base, use your general agricultural knowledge
6. Politely redirect completely off-topic questions back to farming
7. Always give practical, actionable advice
8. Do NOT use markdown headers (# ##) — just plain text, bold, and bullets
9. Respond in the same language the user writes in (English or Hindi mix is fine)
"""


# ─────────────────────────────────────────────────────────────────────────────
#  ══════════════════════  GEMINI ENGINE  ══════════════════════
# ─────────────────────────────────────────────────────────────────────────────

_gemini_client = None
_gemini_ready  = False
_gemini_error  = None

# Each model has its own separate daily quota pool
_GEMINI_MODELS = [
    "gemini-2.5-flash",               # newest — best responses
    "gemini-2.0-flash",               # reliable workhorse
    "gemini-2.0-flash-lite",          # lightweight — separate quota
]


def _init_gemini() -> bool:
    global _gemini_client, _gemini_ready, _gemini_error

    if _gemini_ready:
        return True

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        _gemini_error = "no_key"
        return False

    try:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        _gemini_ready  = True
        print("[OK] Gemini AI initialized.")
        return True
    except ImportError:
        _gemini_error = "no_package"
        print("[WARN] google-genai not installed.")
        return False
    except Exception as e:
        _gemini_error = str(e)
        print(f"[WARN] Gemini init failed: {e}")
        return False


# Per-session chat history for Gemini
_gemini_sessions: dict = {}


def _try_gemini(user_message: str, session_id: str) -> dict | None:
    """
    Try all Gemini models in sequence.
    Returns response dict on success, None if all models fail (quota/error).
    """
    if not _init_gemini():
        return None

    from google.genai import types

    history = _gemini_sessions.get(session_id, [])

    for model_name in _GEMINI_MODELS:
        try:
            chat = _gemini_client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=512,
                ),
                history=history,
            )
            response = chat.send_message(user_message)
            reply    = response.text.strip()

            _gemini_sessions[session_id] = chat.get_history()[-20:]

            if model_name != _GEMINI_MODELS[0]:
                print(f"[INFO] Gemini fallback used: {model_name}")

            return {"success": True, "response": reply, "powered_by": "gemini", "model": model_name}

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                print(f"[WARN] Gemini {model_name} quota hit, trying next...")
                continue
            else:
                print(f"[ERROR] Gemini {model_name}: {err[:120]}")
                return None   # non-quota error — don't try more models

    print("[WARN] All Gemini models quota exhausted. Trying OpenRouter...")
    return None


def _try_gemini_insight(prompt: str) -> str | None:
    """Try all Gemini models for a one-shot insight generation."""
    if not _init_gemini():
        return None

    from google.genai import types

    for model_name in _GEMINI_MODELS:
        try:
            response = _gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.6,
                    max_output_tokens=200,
                ),
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                continue
            break

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  ══════════════════════  OPENROUTER ENGINE  ══════════════════════
# ─────────────────────────────────────────────────────────────────────────────

_OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_HEADERS = {
    "Content-Type":  "application/json",
    "HTTP-Referer":  "https://agreeculture.app",   # your app name (can be anything)
    "X-Title":       "AgreeCulture Smart Farming",
}

# Free models available on OpenRouter (verified live — May 2026)
# See full list: https://openrouter.ai/models?q=free
_OPENROUTER_MODELS = [
    "openai/gpt-oss-120b:free",                       # OpenAI OSS 120B — large & capable
    "nvidia/nemotron-3-super-120b-a12b:free",         # NVIDIA Nemotron 120B
    "openai/gpt-oss-20b:free",                        # OpenAI OSS 20B — fast
    "minimax/minimax-m2.5:free",                      # MiniMax M2.5 — reliable
    "liquid/lfm-2.5-1.2b-instruct:free",             # Liquid LFM — ultra-fast fallback
    "openrouter/free",                                 # OpenRouter auto-routing (last resort)
]

# Per-session chat history for OpenRouter
_openrouter_sessions: dict = {}


def _is_openrouter_configured() -> bool:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return bool(key) and key != "your_openrouter_api_key_here"


def _try_openrouter(user_message: str, session_id: str) -> dict | None:
    """
    Try OpenRouter free models in sequence.
    Returns response dict on success, None if all fail.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key == "your_openrouter_api_key_here":
        return None

    # Build message history for this session
    history = _openrouter_sessions.get(session_id, [])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + \
               [{"role": "user", "content": user_message}]

    headers = {**_OPENROUTER_HEADERS, "Authorization": f"Bearer {api_key}"}

    for model_name in _OPENROUTER_MODELS:
        try:
            payload = {
                "model":      model_name,
                "messages":   messages,
                "temperature": 0.7,
                "max_tokens":  512,
            }
            resp = requests.post(
                _OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=20,
            )

            if resp.status_code == 429:
                print(f"[WARN] OpenRouter {model_name} quota/rate limit, trying next...")
                continue

            if resp.status_code != 200:
                print(f"[WARN] OpenRouter {model_name} error {resp.status_code}: {resp.text[:100]}")
                continue

            data  = resp.json()
            reply = data["choices"][0]["message"]["content"].strip()

            # Update history for this session (keep last 10 exchanges)
            new_history = history + [
                {"role": "user",      "content": user_message},
                {"role": "assistant", "content": reply},
            ]
            _openrouter_sessions[session_id] = new_history[-20:]

            print(f"[INFO] OpenRouter responded via {model_name}")
            return {"success": True, "response": reply, "powered_by": "openrouter", "model": model_name}

        except requests.exceptions.Timeout:
            print(f"[WARN] OpenRouter {model_name} timed out.")
            continue
        except Exception as e:
            print(f"[ERROR] OpenRouter {model_name}: {e}")
            continue

    print("[WARN] All OpenRouter models failed.")
    return None


def _try_openrouter_insight(prompt: str) -> str | None:
    """Try OpenRouter for one-shot insight generation."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key == "your_openrouter_api_key_here":
        return None

    headers  = {**_OPENROUTER_HEADERS, "Authorization": f"Bearer {api_key}"}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    for model_name in _OPENROUTER_MODELS:
        try:
            payload = {"model": model_name, "messages": messages,
                       "temperature": 0.6, "max_tokens": 200}
            resp = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  ══════════════════════  PUBLIC API  ══════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def is_gemini_available() -> bool:
    return _init_gemini()


def get_gemini_status() -> dict:
    ready = _init_gemini()
    return {
        "available":            ready,
        "error":                _gemini_error if not ready else None,
        "openrouter_available": _is_openrouter_configured(),
    }


def get_gemini_response(user_message: str, session_id: str = "default") -> dict:
    """
    Main chat function.
    Tries: Gemini (4 models) -> OpenRouter (3 free models) -> fails to local
    """
    # 1. Try Gemini
    result = _try_gemini(user_message, session_id)
    if result:
        return result

    # 2. Try OpenRouter
    result = _try_openrouter(user_message, session_id)
    if result:
        return result

    # 3. Both exhausted
    return {"success": False, "response": None, "powered_by": "none", "error": "all_quota_exceeded"}


def get_crop_insight(crop_name: str, features: dict) -> str | None:
    """
    Generate personalized farming insight after crop prediction.
    Tries Gemini first, then OpenRouter.
    """
    if not _init_gemini() and not _is_openrouter_configured():
        return None

    prompt = (
        f"A farmer's soil and climate readings: "
        f"Nitrogen={features.get('nitrogen')} kg/ha, "
        f"Phosphorus={features.get('phosphorus')} kg/ha, "
        f"Potassium={features.get('potassium')} kg/ha, "
        f"Temperature={features.get('temperature')}C, "
        f"Humidity={features.get('humidity')}%, "
        f"Rainfall={features.get('rainfall')} mm. "
        f"Our ML model recommends growing {crop_name}. "
        f"Give a 2-3 sentence personalized farming tip based on these specific values. "
        f"Be practical and specific to the numbers. Use 1-2 emojis."
    )

    return _try_gemini_insight(prompt) or _try_openrouter_insight(prompt)
