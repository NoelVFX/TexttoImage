import urllib.parse
from pathlib import Path
from typing import Tuple

import requests


POLLINATIONS_BASE_URL = "https://image.pollinations.ai/p"
DEFAULT_MODEL = "gpt-image-large"
SUPPORTED_ASPECT_RATIOS = {
    "1024x1024": (1024, 1024),
    "1792x1024": (1792, 1024),
}


class ImageGenerationError(RuntimeError):
    """Raised when Pollinations does not return a usable image."""


def build_pollinations_url(
    prompt_text: str,
    *,
    model_choice: str = DEFAULT_MODEL,
    width: int = 1024,
    height: int = 1024,
) -> str:
    encoded_prompt = urllib.parse.quote(prompt_text)
    return (
        f"{POLLINATIONS_BASE_URL}/{encoded_prompt}"
        f"?width={width}"
        f"&height={height}"
        f"&model={urllib.parse.quote(model_choice)}"
        f"&nologo=true"
    )


def generate_pollinations_image_bytes(
    prompt_text: str,
    *,
    model_choice: str = DEFAULT_MODEL,
    width: int = 1024,
    height: int = 1024,
    timeout: int = 60,
) -> Tuple[bytes, str]:
    """Generate an image and return (bytes, content_type)."""
    if not prompt_text or not prompt_text.strip():
        raise ValueError("Prompt is required.")

    gateway_url = build_pollinations_url(
        prompt_text.strip(),
        model_choice=model_choice,
        width=width,
        height=height,
    )
    response = requests.get(gateway_url, timeout=timeout)
    content_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0]

    if response.status_code != 200:
        raise ImageGenerationError(
            f"Image service returned HTTP {response.status_code}: {response.text[:300]}"
        )
    if not content_type.startswith("image/"):
        raise ImageGenerationError(
            f"Image service returned non-image content ({content_type}): {response.text[:300]}"
        )

    return response.content, content_type


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
