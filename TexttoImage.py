import os
import urllib.parse
from pathlib import Path
from typing import Tuple

import requests


def _load_local_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env(Path(__file__).resolve().parent / ".env")

POLLINATIONS_BASE_URL = "https://image.pollinations.ai/p"
# Pollinations' GPT image model can hit the public queue/rate-limit quickly.
# Default to Flux for reliable app demos; set POLLINATIONS_MODEL=gpt-image-large
# or POLLINATIONS_TOKEN for authenticated/premium use.
DEFAULT_MODEL = os.getenv("POLLINATIONS_MODEL", "flux")
DEFAULT_FALLBACK_MODELS = os.getenv("POLLINATIONS_FALLBACK_MODELS", "turbo,gpt-image-large")
SUPPORTED_ASPECT_RATIOS = {
    "1024x1024": (1024, 1024),
    "1792x1024": (1792, 1024),
}


class ImageGenerationError(RuntimeError):
    """Raised when Pollinations does not return a usable image."""


def _pollinations_token() -> str | None:
    _load_local_env(Path(__file__).resolve().parent / ".env")
    token = os.getenv("POLLINATIONS_TOKEN") or os.getenv("POLLINATIONS_API_TOKEN")
    return token.strip() if token and token.strip() else None


def _fallback_models() -> list[str]:
    models = [item.strip() for item in DEFAULT_FALLBACK_MODELS.split(",") if item.strip()]
    return models or ["turbo"]


def build_pollinations_url(
    prompt_text: str,
    *,
    model_choice: str = DEFAULT_MODEL,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    token: str | None = None,
) -> str:
    encoded_prompt = urllib.parse.quote(prompt_text)
    query = {
        "width": str(width),
        "height": str(height),
        "model": model_choice,
        "nologo": "true",
    }
    token = token if token is not None else _pollinations_token()
    if token:
        query["token"] = token
    if seed is not None:
        query["seed"] = str(seed)
    return f"{POLLINATIONS_BASE_URL}/{encoded_prompt}?{urllib.parse.urlencode(query)}"


def _is_pollinations_queue_or_rate_limit(status_code: int, content_type: str, body_preview: str) -> bool:
    normalized = body_preview.lower()
    return (
        status_code in {401, 402, 403, 408, 409, 425, 429, 503}
        or "too many requests" in normalized
        or "request queued" in normalized
        or "queued" in normalized and "unlimited access" in normalized
        or "auth.pollinations.ai" in normalized
    )


def generate_pollinations_image_bytes(
    prompt_text: str,
    *,
    model_choice: str = DEFAULT_MODEL,
    width: int = 1024,
    height: int = 1024,
    timeout: int = 60,
) -> Tuple[bytes, str]:
    """Generate an image and return (bytes, content_type).

    If the selected Pollinations model returns the public queue/rate-limit
    message, retry with lower-friction fallback models before failing.
    """
    if not prompt_text or not prompt_text.strip():
        raise ValueError("Prompt is required.")

    models: list[str] = []
    for model in [model_choice, *_fallback_models()]:
        if model and model not in models:
            models.append(model)

    last_error: ImageGenerationError | None = None
    for model in models:
        gateway_url = build_pollinations_url(
            prompt_text.strip(),
            model_choice=model,
            width=width,
            height=height,
        )
        response = requests.get(gateway_url, timeout=timeout)
        content_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
        body_preview = response.text[:300] if not content_type.startswith("image/") or response.status_code != 200 else ""

        if response.status_code == 200 and content_type.startswith("image/"):
            return response.content, content_type

        if _is_pollinations_queue_or_rate_limit(response.status_code, content_type, body_preview):
            last_error = ImageGenerationError(
                "Pollinations is rate-limiting or queueing this image model. "
                "Set POLLINATIONS_TOKEN from https://auth.pollinations.ai for higher limits, "
                "or use a lighter POLLINATIONS_MODEL such as flux/turbo. "
                f"Last model tried: {model}. Response: {body_preview}"
            )
            continue

        if response.status_code != 200:
            raise ImageGenerationError(
                f"Image service returned HTTP {response.status_code}: {body_preview}"
            )
        raise ImageGenerationError(
            f"Image service returned non-image content ({content_type}): {body_preview}"
        )

    if last_error:
        raise last_error
    raise ImageGenerationError("Pollinations did not return a usable image.")


def generate_pollinations_asset(
    prompt_text: str,
    model_choice: str = DEFAULT_MODEL,
    filename: str = "asset.jpg",
    width: int = 1024,
    height: int = 1024,
) -> str:
    """Generate an image and save it to filename. Returns the saved path."""
    image_bytes, _content_type = generate_pollinations_image_bytes(
        prompt_text,
        model_choice=model_choice,
        width=width,
        height=height,
    )
    output_path = Path(filename)
    output_path.write_bytes(image_bytes)
    return str(output_path)


if __name__ == "__main__":
    text_prompt = "A swimming pool with white walls and clear sky-blue water"
    saved = generate_pollinations_asset(
        text_prompt,
        model_choice=DEFAULT_MODEL,
        filename="output.jpg",
        width=1024,
        height=1024,
    )
    print(f"Saved asset as {saved}")