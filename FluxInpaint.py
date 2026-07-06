from __future__ import annotations

import os
import time
from typing import Any

import requests

FAL_INPAINT_ENDPOINT = os.getenv("FAL_INPAINT_ENDPOINT", "https://queue.fal.run/fal-ai/flux-general/inpainting")
DEFAULT_INFERENCE_STEPS = int(os.getenv("FAL_INPAINT_STEPS", "28"))
DEFAULT_POLL_INTERVAL = float(os.getenv("FAL_INPAINT_POLL_INTERVAL", "1.5"))
DEFAULT_MAX_POLLS = int(os.getenv("FAL_INPAINT_MAX_POLLS", "80"))


class FluxInpaintError(RuntimeError):
    """Raised when FLUX inpainting cannot return a final image URL."""


def fal_api_key() -> str | None:
    key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    return key.strip() if key and key.strip() else None


def build_flux_inpaint_payload(
    *,
    image_url: str,
    mask_url: str,
    prompt: str,
    image_size: str | None = None,
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_url": image_url,
        "mask_url": mask_url,
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
    }
    if image_size:
        payload["image_size"] = image_size
    return payload


def extract_flux_image_url(payload: dict[str, Any]) -> str | None:
    image = payload.get("image")
    if isinstance(image, dict) and image.get("url"):
        return str(image["url"])

    images = payload.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict) and first.get("url"):
            return str(first["url"])
        if isinstance(first, str):
            return first

    data = payload.get("data")
    if isinstance(data, dict):
        return extract_flux_image_url(data)
    return None


def _raise_for_fal_error(response: requests.Response, action: str) -> None:
    if response.status_code < 400:
        return
    preview = response.text.replace("\n", " ")[:300]
    raise FluxInpaintError(f"FLUX inpainting {action} failed with HTTP {response.status_code}: {preview}")


def apply_flux_inpaint(
    *,
    image_url: str,
    mask_url: str,
    prompt: str,
    api_key: str | None = None,
    endpoint: str = FAL_INPAINT_ENDPOINT,
    image_size: str | None = None,
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
    session=requests,
    timeout: int = 60,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    max_polls: int = DEFAULT_MAX_POLLS,
) -> str:
    api_key = api_key or fal_api_key()
    if not api_key:
        raise FluxInpaintError("Set FAL_KEY or FAL_API_KEY to enable FLUX inpainting.")

    payload = build_flux_inpaint_payload(
        image_url=image_url,
        mask_url=mask_url,
        prompt=prompt,
        image_size=image_size,
        num_inference_steps=num_inference_steps,
    )
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    submit_response = session.post(endpoint, json=payload, headers=headers, timeout=timeout)
    _raise_for_fal_error(submit_response, "submit")
    submit_payload = submit_response.json()

    immediate_url = extract_flux_image_url(submit_payload)
    if immediate_url:
        return immediate_url

    response_url = submit_payload.get("response_url") or submit_payload.get("result_url")
    status_url = submit_payload.get("status_url")
    if not response_url and isinstance(submit_payload.get("urls"), dict):
        response_url = submit_payload["urls"].get("get") or submit_payload["urls"].get("response")
        status_url = status_url or submit_payload["urls"].get("status")

    if not response_url:
        raise FluxInpaintError(f"FLUX inpainting did not return image_url or response_url: {submit_payload}")

    for _attempt in range(max_polls):
        if status_url:
            status_response = session.get(status_url, headers=headers, timeout=timeout)
            _raise_for_fal_error(status_response, "status poll")
            status_payload = status_response.json()
            status = str(status_payload.get("status", "")).upper()
            if status in {"FAILED", "ERROR", "CANCELLED"}:
                raise FluxInpaintError(f"FLUX inpainting job failed: {status_payload}")
            if status not in {"COMPLETED", "SUCCESS"}:
                if poll_interval:
                    time.sleep(poll_interval)
                continue

        result_response = session.get(response_url, headers=headers, timeout=timeout)
        if result_response.status_code == 202:
            if poll_interval:
                time.sleep(poll_interval)
            continue
        _raise_for_fal_error(result_response, "result fetch")
        result_payload = result_response.json()
        result_url = extract_flux_image_url(result_payload)
        if result_url:
            return result_url
        if poll_interval:
            time.sleep(poll_interval)

    raise FluxInpaintError("FLUX inpainting timed out before returning a final image URL.")
