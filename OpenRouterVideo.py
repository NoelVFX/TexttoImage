from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import requests


OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_VIDEO_MODEL = "bytedance/seedance-2.0-fast"
SUPPORTED_VIDEO_ASPECT_RATIOS = {
    "1:1": "1:1 square",
    "16:9": "16:9 widescreen",
}
DEFAULT_VIDEO_ASPECT_RATIO = "16:9"
DEFAULT_VIDEO_RESOLUTION = "720p"
DEFAULT_VIDEO_DURATION = 5


class OpenRouterVideoError(RuntimeError):
    """Raised when OpenRouter video generation fails."""


def load_local_env(path: str | Path = ".env") -> None:
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


def get_openrouter_api_key() -> str | None:
    load_local_env(Path(__file__).resolve().parent / ".env")
    return os.getenv("OPENROUTER_API_KEY")


def openrouter_headers() -> Dict[str, str]:
    api_key = get_openrouter_api_key()
    if not api_key:
        raise OpenRouterVideoError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file or deployment environment."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE", "Text to Image and Video Generator")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _raise_for_openrouter_error(response: requests.Response) -> None:
    if response.status_code < 400:
        return
    try:
        error_payload = response.json()
    except ValueError:
        error_payload = {"error": response.text[:500]}
    message = error_payload.get("error", error_payload)
    if isinstance(message, dict):
        message = message.get("message") or message.get("error") or str(message)
    raise OpenRouterVideoError(f"OpenRouter returned HTTP {response.status_code}: {message}")


def submit_video_job(
    prompt: str,
    *,
    aspect_ratio: str = DEFAULT_VIDEO_ASPECT_RATIO,
    duration: int = DEFAULT_VIDEO_DURATION,
    resolution: str = DEFAULT_VIDEO_RESOLUTION,
    generate_audio: bool = False,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Submit a Seedance 2.0 Fast text-to-video job to OpenRouter."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt is required.")
    if aspect_ratio not in SUPPORTED_VIDEO_ASPECT_RATIOS:
        aspect_ratio = DEFAULT_VIDEO_ASPECT_RATIO

    payload: Dict[str, Any] = {
        "model": OPENROUTER_VIDEO_MODEL,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "duration": duration,
        "generate_audio": generate_audio,
    }

    response = requests.post(
        f"{OPENROUTER_API_BASE}/videos",
        headers=openrouter_headers(),
        json=payload,
        timeout=timeout,
    )
    _raise_for_openrouter_error(response)
    return response.json()


def get_video_status(job_id: str, *, timeout: int = 30) -> Dict[str, Any]:
    """Fetch the current status for an OpenRouter video generation job."""
    if not job_id or not job_id.strip():
        raise ValueError("Job id is required.")

    response = requests.get(
        f"{OPENROUTER_API_BASE}/videos/{job_id.strip()}",
        headers=openrouter_headers(),
        timeout=timeout,
    )
    _raise_for_openrouter_error(response)
    return response.json()


def extract_video_url(status_payload: Dict[str, Any]) -> str | None:
    """Return the first downloadable video URL from a completed status payload."""
    unsigned_urls = status_payload.get("unsigned_urls") or []
    if unsigned_urls:
        return unsigned_urls[0]

    job_id = status_payload.get("id")
    if status_payload.get("status") == "completed" and job_id:
        return f"{OPENROUTER_API_BASE}/videos/{job_id}/content?index=0"
    return None