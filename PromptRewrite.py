from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import requests

from OpenRouterVideo import OPENROUTER_API_BASE, load_local_env, openrouter_headers


DEFAULT_PROMPT_REWRITE_TIMEOUT = 30
DEFAULT_PROMPT_REWRITE_MAX_WORDS = 45
DEFAULT_PROMPT_REWRITE_PROVIDER = "openai"
DEFAULT_PROMPT_REWRITE_MODEL = "gpt-5-mini"
OPENAI_API_BASE = "https://api.openai.com/v1"
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
    command = [*_hermes_base_command(hermes_command), "chat", "-Q"]
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


def _rewrite_provider() -> str:
    _load_project_env()
    return (os.getenv("PROMPT_REWRITE_PROVIDER") or DEFAULT_PROMPT_REWRITE_PROVIDER).strip().lower()


def _api_key_available() -> bool:
    _load_project_env()
    provider = _rewrite_provider()
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return False


def _hermes_base_command(hermes_command: str | None = None) -> list[str]:
    """HERMES_COMMAND as an argv prefix. Supports multi-token values so the
    Windows-hosted app can call a WSL-installed agent: HERMES_COMMAND=wsl -e hermes.
    A plain path without spaces is kept as a single token (Windows backslashes safe)."""
    _load_project_env()
    raw = (hermes_command or os.getenv("HERMES_COMMAND", "hermes")).strip() or "hermes"
    if " " not in raw:
        return [raw]
    return shlex.split(raw)


def _hermes_cli_available() -> bool:
    return shutil.which(_hermes_base_command()[0]) is not None


def _use_direct_api() -> bool:
    """Decide between the hermes agent CLI and the OpenRouter HTTP API.

    The hermes agent is preferred whenever its binary is installed (local dev,
    a VM with hermes set up) — same behavior as before. Serverless hosts
    (Vercel) have no hermes binary, so there the OpenRouter API is called
    directly with the same provider/model/prompt. Overrides:
    PROMPT_REWRITE_USE_CLI=1 forces the CLI, =0 forces the API.
    """
    _load_project_env()
    forced = os.getenv("PROMPT_REWRITE_USE_CLI")
    if forced == "1":
        return False
    api_possible = _rewrite_provider() in ("openai", "openrouter") and _api_key_available()
    if forced == "0":
        return api_possible
    if _hermes_cli_available():
        return False  # hermes agent installed -> use it, like before
    return api_possible


def rewrite_prompt_via_api(
    prompt: str,
    *,
    media_type: str,
    aspect_ratio: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> str:
    """Rewrite the prompt by calling the provider's chat API directly (no CLI needed).

    PROMPT_REWRITE_PROVIDER=openai -> api.openai.com with OPENAI_API_KEY (default);
    PROMPT_REWRITE_PROVIDER=openrouter -> openrouter.ai with OPENROUTER_API_KEY.
    """
    _load_project_env()
    provider = _rewrite_provider()
    model = model or os.getenv("PROMPT_REWRITE_MODEL") or DEFAULT_PROMPT_REWRITE_MODEL
    timeout = timeout if timeout is not None else max(1, _env_int("PROMPT_REWRITE_TIMEOUT", DEFAULT_PROMPT_REWRITE_TIMEOUT))

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise PromptRewriteError("AI prompt rewrite is not configured: OPENAI_API_KEY is not set.")
        # Accept OpenRouter-style ids (openai/gpt-5-mini) for the native API too.
        model = model.removeprefix("openai/")
        url = f"{OPENAI_API_BASE}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    else:
        url = f"{OPENROUTER_API_BASE}/chat/completions"
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
    }
    if provider == "openai" and model.startswith(("gpt-5", "o")):
        # Reasoning models reject max_tokens/temperature; cap generously because
        # max_completion_tokens includes hidden reasoning tokens.
        payload["max_completion_tokens"] = _env_int("PROMPT_REWRITE_MAX_COMPLETION_TOKENS", 2000)
        effort = os.getenv("PROMPT_REWRITE_REASONING_EFFORT", "minimal").strip()
        if effort:
            payload["reasoning_effort"] = effort
    else:
        payload["max_tokens"] = 200
        payload["temperature"] = 0.7

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
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
        key_name = "OPENAI_API_KEY" if _rewrite_provider() == "openai" else "OPENROUTER_API_KEY"
        raise PromptRewriteError(
            f"AI prompt rewrite failed: the hermes CLI is not installed and {key_name} is not set. "
            f"Set {key_name} (recommended for serverless) or install the hermes CLI."
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
