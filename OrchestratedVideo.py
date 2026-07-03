from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from OpenRouterVideo import load_local_env
from TexttoImage import DEFAULT_FALLBACK_MODELS, DEFAULT_MODEL, build_pollinations_url


DEFAULT_MAX_FRAME_ATTEMPTS = 1
# Default Hermes visual-review budget. Railway can override this with
# HERMES_REVIEW_TIMEOUT when balancing strict review vs request latency.
DEFAULT_HERMES_TIMEOUT = 30
DEFAULT_HERMES_REVIEW_PROVIDER = "openrouter"
DEFAULT_HERMES_REVIEW_MODEL = "openai/gpt-4o-mini"
DEFAULT_REJECTION_CONFIDENCE_THRESHOLD = 0.85
MIN_IMAGE_BYTES = 2048
IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class VideoOrchestrationError(RuntimeError):
    """Raised when the free storyboard frame should not be sent to paid I2V."""


@dataclass(frozen=True)
class VisionCritique:
    approved: bool
    confidence: float
    reason: str
    improvements: list[str]
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FirstFrameResult:
    original_prompt: str
    optimized_prompt: str
    start_frame_url: str
    critique: VisionCritique
    attempts: int
    width: int
    height: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["vision_critique"] = payload.pop("critique")
        return payload


def _load_project_env() -> None:
    load_local_env(Path(__file__).resolve().parent / ".env")


def _env_flag(name: str, default: str = "0") -> bool:
    _load_project_env()
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    _load_project_env()
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    _load_project_env()
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _strict_review_enabled() -> bool:
    return _env_flag("VIDEO_ORCHESTRATOR_STRICT_REVIEW", "0")


def _soft_review_failures_enabled() -> bool:
    return _env_flag("VIDEO_ORCHESTRATOR_SOFT_REVIEW_FAILURES", "1") and not _strict_review_enabled()


def _extract_text(response: Any) -> str:
    text = getattr(response, "stdout", None)
    if text:
        return str(text).strip()
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    return str(response).strip()


def stable_seed(text: str) -> int:
    """Return a deterministic Pollinations seed for a prompt/ratio combination."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _normalise_improvements(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _critique_from_json(raw: str) -> VisionCritique:
    parsed = _parse_json_object(raw)
    if not parsed:
        reason = "Vision Agent returned an unreadable critique."
        if _soft_review_failures_enabled():
            return VisionCritique(
                approved=True,
                confidence=0.25,
                reason=f"{reason} Soft review mode approved the structurally valid storyboard instead of blocking I2V.",
                improvements=["For stricter blocking, set VIDEO_ORCHESTRATOR_STRICT_REVIEW=true."],
                raw_response=raw,
            )
        return VisionCritique(
            approved=False,
            confidence=0.0,
            reason=f"{reason} The paid I2V job was blocked because strict review is enabled.",
            improvements=["Retry the Hermes visual review and require strict JSON output."],
            raw_response=raw,
        )

    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    critique = VisionCritique(
        approved=bool(parsed.get("approved")),
        confidence=confidence,
        reason=str(parsed.get("reason") or "No reason supplied."),
        improvements=_normalise_improvements(parsed.get("improvements")),
        raw_response=raw,
    )
    return _apply_soft_review_policy(critique)


def _apply_soft_review_policy(critique: VisionCritique) -> VisionCritique:
    """Keep the review useful without letting low-confidence nitpicks block every I2V request."""
    if critique.approved or _strict_review_enabled():
        return critique

    threshold = max(
        0.0,
        min(
            1.0,
            _env_float("VIDEO_ORCHESTRATOR_REJECTION_CONFIDENCE_THRESHOLD", DEFAULT_REJECTION_CONFIDENCE_THRESHOLD),
        ),
    )
    if critique.confidence >= threshold:
        return critique

    return VisionCritique(
        approved=True,
        confidence=critique.confidence,
        reason=(
            "Soft review mode approved a low-confidence Vision Agent rejection "
            f"({critique.confidence:.2f} < {threshold:.2f}): {critique.reason}"
        ),
        improvements=critique.improvements,
        raw_response=critique.raw_response,
    )


def build_hermes_review_prompt(*, user_intent: str, optimized_prompt: str, aspect_ratio: str) -> str:
    return f"""
You are the Vision Agent reviewer for a cost-saving image-to-video pipeline.
Compare the attached storyboard image to the user's requested video. Decide if this exact image is usable as a first frame for a paid I2V API.

User intent: {user_intent}
Storyboard prompt: {optimized_prompt}
Target aspect ratio: {aspect_ratio}

Use a permissive approve-by-default policy. Approve when the main subject and composition are recognizable enough for animation, even if there are minor style differences, small prompt omissions, or ordinary AI-image imperfections.

Reject only for severe, objective problems: wrong main subject, non-image/blank frame, unreadable or heavily smeared composition, gross stretching/warping, badly malformed main anatomy/objects, or broken UI/game asset geometry that would make the video unusable.

If you are unsure, approve with lower confidence and list improvements instead of blocking the I2V job.

Return compact JSON only with this exact schema and no markdown:
{{"approved": true/false, "confidence": 0.0-1.0, "reason": "short reason", "improvements": ["fix 1", "fix 2"]}}
""".strip()


def build_hermes_review_command(
    *,
    image_path: str,
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    hermes_command: str | None = None,
) -> list[str]:
    _load_project_env()
    provider = provider or os.getenv("HERMES_REVIEW_PROVIDER") or DEFAULT_HERMES_REVIEW_PROVIDER
    model = model or os.getenv("HERMES_REVIEW_MODEL") or DEFAULT_HERMES_REVIEW_MODEL
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
            "--image",
            image_path,
            "-q",
            prompt,
        ]
    )
    return command


def _run_hermes_text_prompt(
    prompt: str,
    *,
    runner: Callable[..., Any] = subprocess.run,
    timeout: int = DEFAULT_HERMES_TIMEOUT,
) -> str | None:
    _load_project_env()
    command = [os.getenv("HERMES_COMMAND", "hermes"), "chat", "-Q", "--ignore-rules", "--ignore-user-config", "--source", "tool", "--max-turns", "1"]
    provider = os.getenv("HERMES_REVIEW_PROVIDER") or DEFAULT_HERMES_REVIEW_PROVIDER
    model = os.getenv("HERMES_REVIEW_MODEL") or DEFAULT_HERMES_REVIEW_MODEL
    if provider:
        command.extend(["--provider", provider])
    if model:
        command.extend(["-m", model])
    command.extend(["-q", prompt])

    try:
        result = runner(command, text=True, capture_output=True, timeout=timeout, check=False)
    except Exception:
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    text = _extract_text(result)
    return text or None


def optimize_prompt_for_storyboard(
    user_intent: str,
    *,
    aspect_ratio: str,
    runner: Callable[..., Any] = subprocess.run,
    model: str | None = None,
    client: Any | None = None,
) -> str:
    """Expand the user's prompt into a visual first-frame prompt.

    By default this uses a deterministic local optimizer. Set
    VIDEO_ORCHESTRATOR_PROMPT_OPTIMIZER=hermes to ask the local Hermes Agent
    CLI to do the text-only prompt optimization through its configured model.
    """
    cleaned = user_intent.strip()
    if not cleaned:
        raise ValueError("Prompt is required.")

    _load_project_env()
    optimizer = os.getenv("VIDEO_ORCHESTRATOR_PROMPT_OPTIMIZER", "local").strip().lower()
    if optimizer == "hermes":
        prompt = f"""
You are Hermes, an expert prompt engineer for text-to-image and image-to-video workflows.
Expand the user's intent into one visually descriptive first-frame prompt.
Keep perspective, scale, and object proportions consistent. Avoid wording that causes horizontal stretching, warped anatomy, duplicated UI elements, fisheye distortion, or broken compositions.
Do not include aspect-ratio labels or dimensions. Return only the improved prompt.

User intent: {cleaned}
Target aspect ratio: {aspect_ratio}
""".strip()
        optimized = _run_hermes_text_prompt(prompt, runner=runner)
        if optimized:
            return optimized

    return (
        f"Static cinematic first frame for image-to-video: {cleaned}. "
        "Clean centered composition, consistent perspective, natural proportions, crisp subject silhouette, "
        "balanced negative space, no stretching, no warping, no distorted edges, no duplicated or broken details."
    )


def _build_revision_prompt(previous_prompt: str, critique: VisionCritique) -> str:
    fixes = "; ".join(critique.improvements) or critique.reason
    return (
        f"{previous_prompt}. Regenerate with these corrections: {fixes}. "
        "Preserve the user's subject while fixing composition, proportions, and prompt alignment."
    )


def _fallback_structural_critique(
    image_bytes: bytes,
    content_type: str,
    *,
    strict_reason: str | None = None,
) -> VisionCritique:
    if strict_reason:
        return VisionCritique(
            approved=False,
            confidence=0.0,
            reason=strict_reason,
            improvements=["Install/configure Hermes Agent with a vision-capable model, or disable strict visual review for local development."],
            raw_response="",
        )
    if not content_type.startswith("image/"):
        return VisionCritique(
            approved=False,
            confidence=0.1,
            reason=f"Storyboard service returned non-image content: {content_type}",
            improvements=["Regenerate a valid image storyboard frame before starting I2V."],
            raw_response="",
        )
    if len(image_bytes) < MIN_IMAGE_BYTES:
        return VisionCritique(
            approved=False,
            confidence=0.2,
            reason="Storyboard image payload is too small to trust as a complete frame.",
            improvements=["Regenerate a complete, high-quality storyboard image."],
            raw_response="",
        )
    return VisionCritique(
        approved=True,
        confidence=0.55,
        reason="Visual review is not configured; passed structural safety checks for image content and size.",
        improvements=[],
        raw_response="",
    )


def _temp_image_path(content_type: str, temp_dir: Path | None = None) -> Path:
    suffix = IMAGE_EXTENSIONS.get(content_type.lower(), ".img")
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(prefix="hermes-storyboard-", suffix=suffix, dir=temp_dir, delete=False)
    else:
        handle = tempfile.NamedTemporaryFile(prefix="hermes-storyboard-", suffix=suffix, delete=False)
    path = Path(handle.name)
    handle.close()
    return path


def _critique_with_hermes_cli(
    image_bytes: bytes,
    content_type: str,
    *,
    user_intent: str,
    optimized_prompt: str,
    aspect_ratio: str,
    runner: Callable[..., Any] = subprocess.run,
    temp_dir: Path | None = None,
    timeout: int | None = None,
) -> VisionCritique:
    _load_project_env()
    timeout = timeout if timeout is not None else max(1, _env_int("HERMES_REVIEW_TIMEOUT", DEFAULT_HERMES_TIMEOUT))
    image_path = _temp_image_path(content_type, temp_dir=temp_dir)
    try:
        image_path.write_bytes(image_bytes)
        review_prompt = build_hermes_review_prompt(
            user_intent=user_intent,
            optimized_prompt=optimized_prompt,
            aspect_ratio=aspect_ratio,
        )
        command = build_hermes_review_command(
            image_path=str(image_path),
            prompt=review_prompt,
            provider=os.getenv("HERMES_REVIEW_PROVIDER") or None,
            model=os.getenv("HERMES_REVIEW_MODEL") or None,
        )
        result = runner(command, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        if _env_flag("VIDEO_ORCHESTRATOR_ALLOW_REVIEW_TIMEOUT", "0") or _soft_review_failures_enabled():
            return VisionCritique(
                approved=True,
                confidence=0.35,
                reason=(
                    f"Hermes visual review timed out after {timeout} seconds; proceeding because "
                    "soft review mode is enabled and the storyboard passed structural checks."
                ),
                improvements=["Use a faster vision-review model or raise HERMES_REVIEW_TIMEOUT for stricter review."],
                raw_response=str(exc),
            )
        return VisionCritique(
            approved=False,
            confidence=0.0,
            reason=f"Hermes visual review timed out after {timeout} seconds, so the paid I2V job was blocked.",
            improvements=["Set HERMES_REVIEW_TIMEOUT higher, choose a faster review model, enable VIDEO_ORCHESTRATOR_SOFT_REVIEW_FAILURES, or enable VIDEO_ORCHESTRATOR_ALLOW_REVIEW_TIMEOUT."],
            raw_response=str(exc),
        )
    except Exception as exc:
        if _soft_review_failures_enabled():
            return VisionCritique(
                approved=True,
                confidence=0.25,
                reason=(
                    f"Hermes visual review failed ({exc}); soft review mode approved the "
                    "structurally valid storyboard instead of blocking I2V."
                ),
                improvements=["Check that the hermes CLI is installed and configured with a vision-capable model."],
                raw_response="",
            )
        return VisionCritique(
            approved=False,
            confidence=0.0,
            reason=f"Hermes visual review failed, so the paid I2V job was blocked: {exc}",
            improvements=["Check that the hermes CLI is installed and configured with a vision-capable model."],
            raw_response="",
        )
    finally:
        try:
            image_path.unlink()
        except FileNotFoundError:
            pass

    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    raw = stdout.strip() or stderr.strip()
    if getattr(result, "returncode", 1) != 0:
        if _soft_review_failures_enabled():
            return VisionCritique(
                approved=True,
                confidence=0.25,
                reason=(
                    f"Hermes visual review exited with code {getattr(result, 'returncode', 'unknown')}; "
                    "soft review mode approved the structurally valid storyboard instead of blocking I2V."
                ),
                improvements=["Check Hermes model/provider configuration and ensure the selected model supports image input."],
                raw_response=raw,
            )
        return VisionCritique(
            approved=False,
            confidence=0.0,
            reason=f"Hermes visual review exited with code {getattr(result, 'returncode', 'unknown')}, so the paid I2V job was blocked.",
            improvements=["Check Hermes model/provider configuration and ensure the selected model supports image input."],
            raw_response=raw,
        )
    return _critique_from_json(raw)


def critique_storyboard_image(
    image_bytes: bytes,
    content_type: str,
    *,
    user_intent: str,
    optimized_prompt: str,
    aspect_ratio: str,
    reviewer: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    temp_dir: Path | None = None,
    client: Any | None = None,
    model: str | None = None,
) -> VisionCritique:
    """Use a visual AI agent to decide whether a free frame is safe for paid I2V."""
    _load_project_env()
    if not content_type.startswith("image/") or len(image_bytes) < MIN_IMAGE_BYTES:
        return _fallback_structural_critique(image_bytes, content_type)

    selected_reviewer = (reviewer or os.getenv("VIDEO_ORCHESTRATOR_REVIEWER", "hermes")).strip().lower()
    if selected_reviewer == "hermes":
        return _critique_with_hermes_cli(
            image_bytes,
            content_type,
            user_intent=user_intent,
            optimized_prompt=optimized_prompt,
            aspect_ratio=aspect_ratio,
            runner=runner,
            temp_dir=temp_dir,
        )

    require_vision = _env_flag("VIDEO_ORCHESTRATOR_REQUIRE_VISION", "0")
    if require_vision:
        return _fallback_structural_critique(
            image_bytes,
            content_type,
            strict_reason=f"Visual AI review is required, but reviewer '{selected_reviewer}' is not available.",
        )
    return _fallback_structural_critique(image_bytes, content_type)


def _storyboard_model_choices(preferred: str) -> list[str]:
    fallback_models = [item.strip() for item in DEFAULT_FALLBACK_MODELS.split(",") if item.strip()]
    choices: list[str] = []
    for model in [preferred, *fallback_models]:
        if model and model not in choices:
            choices.append(model)
    return choices or [preferred]


def download_storyboard_bytes(url: str, *, timeout: int = 8) -> tuple[bytes, str]:
    response = requests.get(url, timeout=timeout)
    content_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
    body_preview = response.text[:300] if response.status_code != 200 or not content_type.startswith("image/") else ""
    normalized = body_preview.lower()
    if (
        response.status_code in {401, 402, 403, 408, 409, 425, 429, 503}
        or "too many requests" in normalized
        or "request queued" in normalized
        or "auth.pollinations.ai" in normalized
    ):
        raise VideoOrchestrationError(
            "Pollinations is rate-limiting or queueing the free storyboard frame. "
            "This usually happens with public GPT image models. Set POLLINATIONS_TOKEN from "
            "https://auth.pollinations.ai for higher limits, or set POLLINATIONS_MODEL=flux/turbo. "
            f"Response: {body_preview}"
        )
    if response.status_code != 200:
        raise VideoOrchestrationError(f"Storyboard image service returned HTTP {response.status_code}: {body_preview}")
    if not content_type.startswith("image/"):
        raise VideoOrchestrationError(f"Storyboard image service returned non-image content ({content_type}): {body_preview}")
    return response.content, content_type


def orchestrate_video_first_frame(
    user_prompt: str,
    *,
    aspect_ratio: str,
    width: int,
    height: int,
    model_choice: str = DEFAULT_MODEL,
    max_attempts: int = DEFAULT_MAX_FRAME_ATTEMPTS,
    prompt_optimizer: Callable[..., str] = optimize_prompt_for_storyboard,
    vision_critic: Callable[..., VisionCritique] = critique_storyboard_image,
    downloader: Callable[..., tuple[bytes, str]] = download_storyboard_bytes,
) -> FirstFrameResult:
    """Create and approve a free storyboard frame before any paid I2V call.

    The function only returns when the Vision Agent approves the frame. If all
    attempts are rejected, it raises VideoOrchestrationError so the caller can
    avoid spending OpenRouter video credits.
    """
    if not user_prompt or not user_prompt.strip():
        raise ValueError("Prompt is required.")
    if max_attempts < 1:
        max_attempts = 1

    optimized_prompt = prompt_optimizer(user_prompt, aspect_ratio=aspect_ratio)
    last_critique: VisionCritique | None = None

    for attempt in range(1, max_attempts + 1):
        last_download_error: VideoOrchestrationError | None = None
        for storyboard_model in _storyboard_model_choices(model_choice):
            seed = stable_seed(f"{optimized_prompt}|{aspect_ratio}|{attempt}|{storyboard_model}")
            start_frame_url = build_pollinations_url(
                optimized_prompt,
                model_choice=storyboard_model,
                width=width,
                height=height,
                seed=seed,
            )
            try:
                image_bytes, content_type = downloader(start_frame_url)
            except VideoOrchestrationError as exc:
                message = str(exc).lower()
                if "rate-limiting" in message or "queueing" in message or "too many requests" in message:
                    last_download_error = exc
                    continue
                raise
            critique = vision_critic(
                image_bytes,
                content_type,
                user_intent=user_prompt,
                optimized_prompt=optimized_prompt,
                aspect_ratio=aspect_ratio,
            )
            last_critique = critique
            if critique.approved:
                return FirstFrameResult(
                    original_prompt=user_prompt,
                    optimized_prompt=optimized_prompt,
                    start_frame_url=start_frame_url,
                    critique=critique,
                    attempts=attempt,
                    width=width,
                    height=height,
                    seed=seed,
                )
            optimized_prompt = _build_revision_prompt(optimized_prompt, critique)
            break
        else:
            if last_download_error:
                raise last_download_error

    reason = last_critique.reason if last_critique else "No critique returned."
    raise VideoOrchestrationError(f"Vision Agent rejected the free storyboard frame: {reason}")
