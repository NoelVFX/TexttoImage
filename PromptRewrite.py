from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

import requests

from OpenRouterVideo import OPENROUTER_API_BASE, load_local_env, openrouter_headers


DEFAULT_PROMPT_REWRITE_TIMEOUT = 30
DEFAULT_PROMPT_REWRITE_MAX_WORDS = 45
DEFAULT_PROMPT_REWRITE_PROVIDER = "openrouter"
DEFAULT_PROMPT_REWRITE_MODEL = "openai/gpt-4o-mini"
SUPPORTED_MEDIA_TYPES = {"image", "video"}
WARNING_LINE_PATTERNS = (
    re.compile(r"tirith security scanner", flags=re.IGNORECASE),
    re.compile(r"command scanning will use pattern matching only", flags=re.IGNORECASE),
)


class PromptRewriteError(RuntimeError):
    """Raised when the AI prompt rewrite step cannot return a usable prompt."""


def _load_project_env() -> None:
    load_local_env(Path(__file__).resolve().parent / ".env")


def _extract_text(response: Any) -> str:
    text = getattr(response, "stdout", None)
    if text:
        return str(text).strip()
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    return str(response).strip()


def _env_int(name: str, default: int) -> int:
    _load_project_env()
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def normalise_media_type(media_type: str | None) -> str:
    selected = (media_type or "image").strip().lower()
    return selected if selected in SUPPORTED_MEDIA_TYPES else "image"


def build_prompt_rewrite_prompt(prompt: str, *, media_type: str, aspect_ratio: str | None = None) -> str:
    selected_media_type = normalise_media_type(media_type)
    ratio = (aspect_ratio or "not specified").strip() or "not specified"
    if selected_media_type == "video":
        media_guidance = (
            "Optimize for text-to-video or image-to-video generation. Include camera movement, subject motion, "
            "lighting, angle/lens, mood, environment, quality, and optional sound cues. Keep it punchy."
        )
    else:
        media_guidance = (
            "Optimize for text-to-image generation. Include composition, camera angle, lighting, mood, "
            "environment, texture, depth of field, and quality. Keep it punchy."
        )

    return f"""
You are an AI prompt engineering gatekeeper for an image/video generation web app.
Rewrite the user's short idea into a richer, production-ready generation prompt.

Media type: {selected_media_type}
Aspect ratio: {ratio}
User prompt: {prompt.strip()}

Requirements:
- Preserve the user's subject and intent exactly.
- Add concrete visual details: lighting, camera angle, lens/composition, cinematic style, mood, environment, textures, and quality.
- Keep it to one short sentence under {DEFAULT_PROMPT_REWRITE_MAX_WORDS} words.
- Avoid policy/safety commentary, explanations, markdown, bullet lists, JSON, labels, or quotation marks.
- Return only the rewritten prompt text.
- Do not include warnings, scanner notices, diagnostics, prefixes, or setup messages.

{media_guidance}
""".strip()


def build_prompt_rewrite_command(
    prompt: str,
    *,
    media_type: str,
    aspect_ratio: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    hermes_command: str | None = None,
) -> list[str]:
    _load_project_env()
    provider = provider or os.getenv("PROMPT_REWRITE_PROVIDER") or DEFAULT_PROMPT_REWRITE_PROVIDER
    model = model or os.getenv("PROMPT_REWRITE_MODEL") or DEFAULT_PROMPT_REWRITE_MODEL
    command = [hermes_command or os.getenv("HERMES_COMMAND", "hermes"), "chat", "-Q"]
    if provider:
        command.extend(["--provider", provider])
    if model:
        command.extend(["-m", model])
    command.extend(
        [
            "--ignore-rules",
            "--ignore-user-config",
            "--source",
            "tool",
            "--max-turns",
            "1",
            "-q",
            build_prompt_rewrite_prompt(prompt, media_type=media_type, aspect_ratio=aspect_ratio),
        ]
    )
    return command


def clean_rewritten_prompt(text: str) -> str:
    cleaned = "\n".join(
        line
        for line in text.strip().splitlines()
        if line.strip() and not any(pattern.search(line) for pattern in WARNING_LINE_PATTERNS)
    ).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    cleaned = re.sub(r"^(rewritten prompt|prompt)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def limit_prompt_words(text: str, max_words: int = DEFAULT_PROMPT_REWRITE_MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    shortened = " ".join(words[:max_words]).rstrip(" ,;:")
    if shortened and shortened[-1] not in ".!?":
        shortened += "."
    return shortened


def _api_key_available() -> bool:
    _load_project_env()
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _use_direct_api() -> bool:
    """Prefer the OpenRouter HTTP API over the hermes CLI when possible.

    Serverless hosts (Vercel) have no hermes binary, so the CLI path fails with
    "no such file or directory: hermes". Direct API is used when the provider is
    OpenRouter and an API key is configured; set PROMPT_REWRITE_USE_CLI=1 to
    force the CLI locally.
    """
    _load_project_env()
    if os.getenv("PROMPT_REWRITE_USE_CLI") == "1":
        return False
    provider = (os.getenv("PROMPT_REWRITE_PROVIDER") or DEFAULT_PROMPT_REWRITE_PROVIDER).strip().lower()
    return provider == "openrouter" and _api_key_available()


def rewrite_prompt_via_api(
    prompt: str,
    *,
    media_type: str,
    aspect_ratio: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> str:
    """Rewrite the prompt by calling OpenRouter's chat API directly (no CLI needed)."""
    _load_project_env()
    model = model or os.getenv("PROMPT_REWRITE_MODEL") or DEFAULT_PROMPT_REWRITE_MODEL
    timeout = timeout if timeout is not None else max(1, _env_int("PROMPT_REWRITE_TIMEOUT", DEFAULT_PROMPT_REWRITE_TIMEOUT))
    try:
        headers = openrouter_headers()
    except Exception as exc:
        raise PromptRewriteError(f"AI prompt rewrite is not configured: {exc}") from exc
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_prompt_rewrite_prompt(prompt, media_type=media_type, aspect_ratio=aspect_ratio),
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    try:
        response = requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise PromptRewriteError(f"AI prompt rewrite request failed: {exc}") from exc
    if response.status_code != 200:
        raise PromptRewriteError(f"AI prompt rewrite returned HTTP {response.status_code}: {response.text[:300]}")
    try:
        raw = response.json()["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise PromptRewriteError(f"AI prompt rewrite returned an unexpected response: {response.text[:300]}") from exc
    rewritten = limit_prompt_words(clean_rewritten_prompt(raw))
    if not rewritten:
        raise PromptRewriteError("AI prompt rewrite returned an empty prompt.")
    return rewritten


def rewrite_prompt(
    prompt: str,
    *,
    media_type: str,
    aspect_ratio: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    timeout: int | None = None,
) -> str:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("Prompt is required.")

    # Direct API path (default in serverless deployments). An injected runner
    # means a caller/test wants the CLI path specifically, so respect it.
    if runner is subprocess.run and _use_direct_api():
        return rewrite_prompt_via_api(
            cleaned_prompt,
            media_type=media_type,
            aspect_ratio=aspect_ratio,
            timeout=timeout,
        )

    timeout = timeout if timeout is not None else max(1, _env_int("PROMPT_REWRITE_TIMEOUT", DEFAULT_PROMPT_REWRITE_TIMEOUT))
    command = build_prompt_rewrite_command(
        cleaned_prompt,
        media_type=media_type,
        aspect_ratio=aspect_ratio,
    )
    try:
        result = runner(command, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise PromptRewriteError(f"AI prompt rewrite timed out after {timeout} seconds.") from exc
    except FileNotFoundError as exc:
        # hermes CLI not installed (e.g. on Vercel): fall back to the HTTP API
        # when a key is available instead of failing outright.
        if _api_key_available():
            return rewrite_prompt_via_api(
                cleaned_prompt,
                media_type=media_type,
                aspect_ratio=aspect_ratio,
                timeout=timeout,
            )
        raise PromptRewriteError(
            "AI prompt rewrite failed: the hermes CLI is not installed and OPENROUTER_API_KEY is not set. "
            "Set OPENROUTER_API_KEY (recommended for serverless) or install the hermes CLI."
        ) from exc
    except Exception as exc:
        raise PromptRewriteError(f"AI prompt rewrite failed: {exc}") from exc

    raw = _extract_text(result)
    if getattr(result, "returncode", 1) != 0:
        raise PromptRewriteError(
            f"AI prompt rewrite exited with code {getattr(result, 'returncode', 'unknown')}: {raw[:300]}"
        )

    rewritten = limit_prompt_words(clean_rewritten_prompt(raw))
    if not rewritten:
        raise PromptRewriteError("AI prompt rewrite returned an empty prompt.")
    return rewritten
