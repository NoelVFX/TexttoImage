from __future__ import annotations

from typing import Any

from TexttoImage import DEFAULT_MODEL, build_pollinations_url


MIN_PATCH_SIZE = 1
MAX_PATCH_SIZE = 2048


class ImageEditError(RuntimeError):
    """Raised when a masked image edit request is invalid."""


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def normalise_mask_box(mask: dict[str, Any], *, image_width: int | None = None, image_height: int | None = None) -> dict[str, int]:
    image_width = image_width or MAX_PATCH_SIZE
    image_height = image_height or MAX_PATCH_SIZE
    x = max(0, _to_int(mask.get("x"), 0))
    y = max(0, _to_int(mask.get("y"), 0))
    width = max(MIN_PATCH_SIZE, _to_int(mask.get("width"), MIN_PATCH_SIZE))
    height = max(MIN_PATCH_SIZE, _to_int(mask.get("height"), MIN_PATCH_SIZE))

    if x >= image_width:
        x = max(0, image_width - MIN_PATCH_SIZE)
    if y >= image_height:
        y = max(0, image_height - MIN_PATCH_SIZE)
    width = min(width, max(MIN_PATCH_SIZE, image_width - x), MAX_PATCH_SIZE)
    height = min(height, max(MIN_PATCH_SIZE, image_height - y), MAX_PATCH_SIZE)
    return {"x": x, "y": y, "width": width, "height": height}


def build_masked_region_prompt(*, micro_prompt: str, context_prompt: str | None = None) -> str:
    context = (context_prompt or "the original image").strip()
    return (
        f"{micro_prompt.strip()}, isolated replacement patch for the masked region only, "
        f"same object scale and proportions as the selected element, match surrounding grass, shadows, texture, "
        f"perspective, color temperature, and lighting from {context}, preserve original background, "
        "background unchanged outside the box, seamless edges, no shrinking, no style mismatch, high detail"
    )


def build_masked_region_edit(
    *,
    image_url: str,
    micro_prompt: str,
    mask: dict[str, Any],
    context_prompt: str | None = None,
    model_choice: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    if not image_url or not str(image_url).strip():
        raise ValueError("Image URL is required.")
    if not micro_prompt or not micro_prompt.strip():
        raise ValueError("A micro-prompt is required for the masked region.")

    image_width = _to_int(mask.get("image_width"), MAX_PATCH_SIZE)
    image_height = _to_int(mask.get("image_height"), MAX_PATCH_SIZE)
    box = normalise_mask_box(mask, image_width=image_width, image_height=image_height)
    patch_prompt = build_masked_region_prompt(micro_prompt=micro_prompt, context_prompt=context_prompt)
    patch_url = build_pollinations_url(
        patch_prompt,
        model_choice=model_choice,
        width=box["width"],
        height=box["height"],
    )
    return {
        "image_url": image_url.strip(),
        "micro_prompt": micro_prompt.strip(),
        "mask": box,
        "patch_prompt": patch_prompt,
        "patch_url": patch_url,
        "workflow": "masked-region-patch-overlay",
    }
