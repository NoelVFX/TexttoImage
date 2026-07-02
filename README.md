# Text to Image and Video Web App

A Flask web app that turns prompts into images with Pollinations and uses a cost-saving Pollinations-to-OpenRouter I2V workflow for videos.

## Features

- Text-to-image prompt input in the browser
- Image aspect ratio selector:
  - `1024x1024` / 1:1 square
  - `1792x1024` / 16:9 widescreen
- Generated image preview on the same page
- Image download link
- Text-to-video prompt input in the browser
- Cost-saving video workflow:
  1. Hermes optimizes the user's prompt for a stable first-frame composition
  2. Build a free static first frame with Pollinations using a deterministic seed
  3. The Vision Agent reviews that exact free frame against the original prompt and rejects off-prompt, stretched, warped, or broken images
  4. Only after approval, pass that image URL into OpenRouter as `frame_images[0]` / `first_frame`
  5. Animate it with OpenRouter model `alibaba/wan-2.6`
- Video aspect ratio selector:
  - `16:9` widescreen
  - `9:16` vertical
- Async video job polling, video preview, and video download link on the same page

## Environment

Video generation requires an OpenRouter API key. The free-frame visual review uses your local Hermes Agent CLI, so Hermes must be installed and configured with a vision-capable model/provider.

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
# verify Hermes is available and configured:
hermes doctor
```

`.env` is ignored by git. Do not commit your real API keys.

Optional variables:

- `OPENROUTER_HTTP_REFERER` - your deployed site URL
- `OPENROUTER_APP_TITLE` - app title shown to OpenRouter
- `VIDEO_ORCHESTRATOR_REVIEWER=hermes` - uses Hermes Agent as the visual reviewer before paid I2V
- `HERMES_COMMAND` - path/name of the Hermes executable; default `hermes`
- `HERMES_REVIEW_PROVIDER` and `HERMES_REVIEW_MODEL` - provider/model override for the Hermes review subprocess; defaults are `openrouter` and `openai/gpt-4o-mini`
- `VIDEO_ORCHESTRATOR_PROMPT_OPTIMIZER=hermes` - optionally asks Hermes to optimize the text prompt too; default `local`
- `VIDEO_ORCHESTRATOR_MAX_ATTEMPTS` - number of free-frame review attempts per request; default `1` for web timeout safety
- `VIDEO_ORCHESTRATOR_REQUIRE_VISION=true` - blocks video generation if the selected visual reviewer is unavailable

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000.

## Deploy

This is a standard Flask app, but the video safety gate shells out to the local `hermes` CLI. For platforms like Render, Railway, Fly.io, or Heroku-style hosts:

- Build command: `pip install -r requirements.txt` (`hermes-agent` is included so the `hermes` CLI is installed during deploy)
- Start command: `gunicorn --timeout 180 app:app` (also pinned in `railway.json` for Railway)
- Required environment variable: `OPENROUTER_API_KEY`
- Required runtime dependency for video review: `hermes` available on `PATH`, or set `HERMES_COMMAND` to its absolute path
- Hermes must be configured with a vision-capable model/provider in that deployment environment

`gunicorn` is already included in `requirements.txt` and `Procfile` is included for Heroku-compatible platforms.

## Push to GitHub

```bash
git init -b main
git add app.py TexttoImage.py OrchestratedVideo.py OpenRouterVideo.py templates/index.html static/styles.css static/generated/.gitkeep test_orchestrated_video.py requirements.txt Procfile railway.json README.md .gitignore .env.example
git commit -m "Build Flask text-to-image and video web app"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## Project files

- `app.py` - Flask routes and web UI/API integration
- `TexttoImage.py` - Pollinations image generation helper
- `OrchestratedVideo.py` - Hermes prompt optimizer, free-frame generation, Vision Agent gate, and retry/blocking logic
- `OpenRouterVideo.py` - OpenRouter Wan 2.6 video generation helper
- `templates/index.html` - Web page template
- `static/styles.css` - Styling
- `static/generated/` - Runtime generated images, ignored by git
- `Procfile` and `railway.json` - deployment start commands
- `test_orchestrated_video.py` - regression tests for the vision-gated I2V workflow
