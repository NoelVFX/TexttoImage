from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from TexttoImage import DEFAULT_MODEL, build_pollinations_url


MIN_PATCH_SIZE = 1
MAX_PATCH_SIZE = 2048

COLOR_NAME_TO_RGB = {
    "red": (220, 38, 38),
    "green": (34, 197, 94),
    "blue": (59, 130, 246),
    "yellow": (234, 179, 8),
    "orange": (249, 115, 22),
    "purple": (168, 85, 247),
    "pink": (236, 72, 153),
    "black": (24, 24, 27),
    "white": (245, 245, 245),
    "brown": (120, 72, 42),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "gold": (234, 179, 8),
    "silver": (180, 180, 190),
}


class ImageEditError(RuntimeError):
    """Raised when a masked image edit request is invalid."""


@dataclass(frozen=True)
class ColorRecolorRequest:
    target_name: str
    target_rgb: tuple[int, int, int]


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


def detect_color_recolor_request(prompt: str) -> ColorRecolorRequest | None:
    """Detect simple masked recolor requests that should not call an image generator.

    For prompts like "change the apple color to green", a generative patch often
    invents a second smaller apple. This path treats it as a deterministic pixel
    operation: preserve the selected object's shape, texture, and shading, and
    only shift its hue.
    """
    normalized = f" {prompt.strip().lower()} "
    if not normalized.strip():
        return None
    if not re.search(r"\b(change|turn|make|recolor|colour|color)\b", normalized):
        return None

    target_name = None
    for color_name in COLOR_NAME_TO_RGB:
        if re.search(rf"\b(to|into|as)\s+{re.escape(color_name)}\b", normalized):
            target_name = color_name
    if target_name is None:
        for color_name in COLOR_NAME_TO_RGB:
            if re.search(rf"\b{re.escape(color_name)}\b", normalized):
                target_name = color_name

    if target_name is None:
        return None
    canonical = "gray" if target_name == "grey" else target_name
    return ColorRecolorRequest(target_name=canonical, target_rgb=COLOR_NAME_TO_RGB[target_name])


def _median_rgb(pixels: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    if not pixels:
        return (0, 0, 0)
    channels = list(zip(*pixels))
    return tuple(int(sorted(channel)[len(channel) // 2]) for channel in channels)  # type: ignore[return-value]


def _rgb_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return sum((left[index] - right[index]) ** 2 for index in range(3)) ** 0.5


def _object_alpha_from_crop(crop: Image.Image) -> Image.Image:
    rgb = crop.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    border_pixels: list[tuple[int, int, int]] = []
    for x in range(width):
        border_pixels.append(pixels[x, 0])
        border_pixels.append(pixels[x, height - 1])
    for y in range(height):
        border_pixels.append(pixels[0, y])
        border_pixels.append(pixels[width - 1, y])

    background_rgb = _median_rgb(border_pixels)
    border_distances = sorted(_rgb_distance(pixel, background_rgb) for pixel in border_pixels)
    background_noise = border_distances[min(len(border_distances) - 1, round(len(border_distances) * 0.9))]
    distance_threshold = max(28, background_noise + 12)

    alpha = Image.new("L", (width, height), 0)
    alpha_pixels = alpha.load()
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            _hue, saturation, _value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
            distance = _rgb_distance((red, green, blue), background_rgb)
            if distance >= distance_threshold and saturation >= 0.12:
                alpha_pixels[x, y] = 255

    return alpha.filter(ImageFilter.MedianFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.2))


def _recolor_crop(crop: Image.Image, target_rgb: tuple[int, int, int]) -> Image.Image:
    rgb = crop.convert("RGB")
    target_hue, target_saturation, _target_value = colorsys.rgb_to_hsv(
        target_rgb[0] / 255,
        target_rgb[1] / 255,
        target_rgb[2] / 255,
    )
    output = Image.new("RGB", rgb.size)
    input_pixels = rgb.load()
    output_pixels = output.load()
    for y in range(rgb.height):
        for x in range(rgb.width):
            red, green, blue = input_pixels[x, y]
            _hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
            new_saturation = max(saturation, min(1.0, target_saturation * 0.82), 0.35)
            nr, ng, nb = colorsys.hsv_to_rgb(target_hue, new_saturation, value)
            output_pixels[x, y] = (round(nr * 255), round(ng * 255), round(nb * 255))
    return output


def recolor_masked_region(
    original_bytes: bytes,
    mask: dict[str, Any],
    target_rgb: tuple[int, int, int],
) -> tuple[bytes, str]:
    original = Image.open(BytesIO(original_bytes)).convert("RGBA")
    box = _scale_mask_box_to_image(mask, image_width=original.width, image_height=original.height)
    crop_box = (box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"])
    crop = original.crop(crop_box).convert("RGB")
    object_alpha = _object_alpha_from_crop(crop)
    recolored_crop = _recolor_crop(crop, target_rgb).convert("RGBA")

    edited = original.copy()
    edited.paste(recolored_crop, (box["x"], box["y"]), object_alpha)
    output = BytesIO()
    edited.convert("RGB").save(output, format="PNG", optimize=True)
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
