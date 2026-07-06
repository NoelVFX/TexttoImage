from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

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
        f"{micro_prompt.strip()}, photorealistic inpaint patch for only the selected rectangle, "
        f"same object scale and proportions as the selected element, match surrounding grass, walls, shadows, texture, "
        f"perspective, color temperature, lens, grain, and lighting from {context}, preserve original background, extend the nearby background "
        "naturally all the way to the patch edges, background unchanged outside the box, seamless edges, "
        "no blue frame, no border, no outline, no shrinking, no style mismatch, high detail"
    )


def _resample_filter():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _scale_mask_box_to_image(mask: dict[str, Any], *, image_width: int, image_height: int) -> dict[str, int]:
    display_width = max(MIN_PATCH_SIZE, _to_int(mask.get("image_width"), image_width))
    display_height = max(MIN_PATCH_SIZE, _to_int(mask.get("image_height"), image_height))
    display_box = normalise_mask_box(mask, image_width=display_width, image_height=display_height)

    scale_x = image_width / display_width
    scale_y = image_height / display_height
    scaled = {
        "x": _to_int(display_box["x"] * scale_x),
        "y": _to_int(display_box["y"] * scale_y),
        "width": max(MIN_PATCH_SIZE, _to_int(display_box["width"] * scale_x)),
        "height": max(MIN_PATCH_SIZE, _to_int(display_box["height"] * scale_y)),
    }
    return normalise_mask_box(scaled, image_width=image_width, image_height=image_height)


def _feather_alpha_mask(width: int, height: int, feather_px: int | None = None) -> Image.Image:
    feather = feather_px if feather_px is not None else max(2, round(min(width, height) * 0.08))
    feather = max(0, min(int(feather), max(0, min(width, height) // 2 - 1)))
    if feather <= 0:
        return Image.new("L", (width, height), 255)

    alpha = Image.new("L", (width + feather * 2, height + feather * 2), 0)
    draw = ImageDraw.Draw(alpha)
    draw.rectangle((feather, feather, width + feather - 1, height + feather - 1), fill=255)
    blurred = alpha.filter(ImageFilter.GaussianBlur(radius=feather))
    cropped = blurred.crop((feather, feather, width + feather, height + feather))
    draw = ImageDraw.Draw(cropped)
    draw.rectangle((feather, feather, width - feather - 1, height - feather - 1), fill=255)
    return cropped


def composite_masked_patch(
    original_bytes: bytes,
    patch_bytes: bytes,
    mask: dict[str, Any],
    *,
    feather_px: int | None = None,
) -> tuple[bytes, str]:
    """Return a single image with the generated patch feather-blended into the original.

    The browser mask is drawn in displayed-image CSS pixels. This scales that box
    back to the real source-image pixels before compositing, so the final PNG can
    replace the displayed image instead of leaving a separate blue-bordered patch
    layer on top of it.
    """
    original = Image.open(BytesIO(original_bytes)).convert("RGBA")
    patch = Image.open(BytesIO(patch_bytes)).convert("RGBA")
    box = _scale_mask_box_to_image(mask, image_width=original.width, image_height=original.height)

    patch = patch.resize((box["width"], box["height"]), _resample_filter())
    alpha = _feather_alpha_mask(box["width"], box["height"], feather_px=feather_px)
    original.paste(patch, (box["x"], box["y"]), alpha)

    output = BytesIO()
    original.convert("RGB").save(output, format="PNG", optimize=True)
    return output.getvalue(), "image/png"


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
