from __future__ import annotations

import mimetypes
import os
import re
import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory, url_for

from TexttoImage import DEFAULT_MODEL, SUPPORTED_ASPECT_RATIOS, generate_pollinations_image_bytes

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def safe_slug(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug[:max_len].strip("-") or "generated-image")


def extension_for_content_type(content_type: str) -> str:
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0])
    if guessed in {".jpe", ".jpeg"}:
        return ".jpg"
    return guessed or ".jpg"


@app.get("/")
def index():
    return render_template(
        "index.html",
        aspect_ratios=SUPPORTED_ASPECT_RATIOS.keys(),
        selected_ratio="1024x1024",
        prompt="",
        image_url=None,
        download_url=None,
        error=None,
    )


@app.post("/generate")
def generate():
    prompt = request.form.get("prompt", "").strip()
    selected_ratio = request.form.get("aspect_ratio", "1024x1024")

    if selected_ratio not in SUPPORTED_ASPECT_RATIOS:
        selected_ratio = "1024x1024"

    if not prompt:
        return render_template(
            "index.html",
            aspect_ratios=SUPPORTED_ASPECT_RATIOS.keys(),
            selected_ratio=selected_ratio,
            prompt=prompt,
            image_url=None,
            download_url=None,
            error="Please enter a prompt before generating an image.",
        ), 400

    width, height = SUPPORTED_ASPECT_RATIOS[selected_ratio]

    try:
        image_bytes, content_type = generate_pollinations_image_bytes(
            prompt,
            model_choice=DEFAULT_MODEL,
            width=width,
            height=height,
        )
    except Exception as exc:  # show useful service errors in the UI
        return render_template(
            "index.html",
            aspect_ratios=SUPPORTED_ASPECT_RATIOS.keys(),
            selected_ratio=selected_ratio,
            prompt=prompt,
            image_url=None,
            download_url=None,
            error=f"Image generation failed: {exc}",
        ), 502

    extension = extension_for_content_type(content_type)
    filename = f"{safe_slug(prompt)}-{selected_ratio}-{uuid.uuid4().hex[:8]}{extension}"
    output_path = GENERATED_DIR / filename
    output_path.write_bytes(image_bytes)

    image_url = url_for("static", filename=f"generated/{filename}")
    download_url = url_for("download_image", filename=filename)

    return render_template(
        "index.html",
        aspect_ratios=SUPPORTED_ASPECT_RATIOS.keys(),
        selected_ratio=selected_ratio,
        prompt=prompt,
        image_url=image_url,
        download_url=download_url,
        error=None,
    )


@app.get("/download/<path:filename>")
def download_image(filename: str):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
