import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()


def get_gemini_api_key() -> str:
    key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env."
        )
    if not key.startswith("AIza"):
        raise RuntimeError(
            "Invalid Gemini API key format. Gemini keys usually start with 'AIza'."
        )
    return key


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _model_candidates() -> list[str]:
    configured = get_gemini_model()
    fallbacks = [
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
    ]
    ordered = [configured] + [m for m in fallbacks if m != configured]
    # Keep order and remove duplicates.
    return list(dict.fromkeys(ordered))


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client(api_key=get_gemini_api_key())


def generate_text(prompt: str) -> str:
    last_error = None
    for model in _model_candidates():
        try:
            response = get_client().models.generate_content(
                model=model,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            return "I could not generate a response. Please try again."
        except errors.ClientError as exc:
            # Retry on model-not-available / not-found cases.
            if getattr(exc, "code", None) == 404:
                last_error = exc
                continue
            raise

    if last_error:
        raise RuntimeError(
            "No available Gemini model for this account. "
            "Set GEMINI_MODEL in .env to an enabled model (for example gemini-2.5-flash)."
        ) from last_error

    raise RuntimeError("Gemini request failed for an unknown reason.")
