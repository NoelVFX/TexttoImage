from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import requests

OPENAI_IMAGE_EDIT_URL = "https://api.openai.com/v1/images/edits"
DEFAULT_OPENAI_IMAGE_EDIT_MODEL = os.getenv("OPENAI_IMAGE_EDIT_MODEL", "gpt-image-1")
DEFAULT_OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "auto")
DEFAULT_OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "auto")


class OpenAIImageEditError(RuntimeError):
    """Raised when OpenAI image editing cannot return an edited image."""


def _load_local_env(path: str | Path | None = None) -> None:
    env_path = Path(path) if path is not None else Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def openai_api_key(*, env_path: str | Path | None = None) -> str | None:
    _load_local_env(env_path)
    for name in ("OPENAI_API_KEY", "OpenAI_API_KEY", "OPENAI_KEY"):
        key = os.getenv(name)
        if key and key.strip():
            return key.strip()
    return None


def build_openai_image_edit_files(image_bytes: bytes, mask_bytes: bytes) -> dict[str, tuple[str, bytes, str]]:
    return {
        "image": ("image.png", image_bytes, "image/png"),
        "mask": ("mask.png", mask_bytes, "image/png"),
    }


def extract_openai_image_bytes(payload: dict[str, Any], *, session=requests, timeout: int = 60) -> bytes:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise OpenAIImageEditError(f"OpenAI image edit returned no image data: {payload}")

    first = data[0]
    if not isinstance(first, dict):
        raise OpenAIImageEditError(f"OpenAI image edit returned malformed image data: {payload}")

    b64_json = first.get("b64_json")
    if b64_json:
        try:
            return base64.b64decode(str(b64_json))
        except Exception as exc:
            raise OpenAIImageEditError("OpenAI image edit returned invalid base64 image data.") from exc

    url = first.get("url")
    if url:
        response = session.get(str(url), timeout=timeout)
        if response.status_code != 200:
            raise OpenAIImageEditError(f"OpenAI image URL download failed with HTTP {response.status_code}: {response.text[:200]}")
        return response.content

    raise OpenAIImageEditError(f"OpenAI image edit response did not include b64_json or url: {payload}")


def apply_openai_image_edit(
    *,
    image_bytes: bytes,
    mask_bytes: bytes,
    prompt: str,
    api_key: str | None = None,
    model: str = DEFAULT_OPENAI_IMAGE_EDIT_MODEL,
    size: str = DEFAULT_OPENAI_IMAGE_SIZE,
    quality: str = DEFAULT_OPENAI_IMAGE_QUALITY,
    session=requests,
    timeout: int = 120,
) -> bytes:
    api_key = api_key or openai_api_key()
    if not api_key:
        raise OpenAIImageEditError("Set OPENAI_API_KEY to enable OpenAI image edits.")

    data = {
        "model": model,
        "prompt": prompt,
    }
    if size:
        data["size"] = size
    if quality:
        data["quality"] = quality

    response = session.post(
        OPENAI_IMAGE_EDIT_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        data=data,
        files=build_openai_image_edit_files(image_bytes, mask_bytes),
        timeout=timeout,
    )
    if response.status_code >= 400:
        preview = response.text.replace("\n", " ")[:300]
        raise OpenAIImageEditError(f"OpenAI image edit failed with HTTP {response.status_code}: {preview}")

    return extract_openai_image_bytes(response.json(), session=session, timeout=timeout)
