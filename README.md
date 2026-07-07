# Text to Image and Video Web App

A Flask web app that turns prompts into images with Pollinations and uses a cost-saving Pollinations-to-OpenRouter I2V workflow for videos.

## Features

- Text-to-image prompt input in the browser
- AI prompt rewrite buttons for image and video prompts, adding lighting, camera angle, cinematic style, detail, and quality terms before generation
- Image aspect ratio selector:
  - `1024x1024` / 1:1 square
  - `1792x1024` / 16:9 widescreen
- Generated image preview on the same page
- OpenAI image-edit masked edits: draw a box over a specific element, type a micro-prompt, and the app uploads the source image plus alpha mask so OpenAI edits only the selected area while preserving unmasked pixels
- Optional FLUX/Fal.ai inpainting masked edits remain available via `INPAINT_PROVIDER=fal`
- Color-only masked edits also use OpenAI image edits with a stricter prompt that asks to preserve object shape, size, texture, lighting, shadows, and background while changing only the selected color
- User auth API backed by MongoDB: register, login, logout, current user, and per-user generation history
- Billing/credits page for logged-in users showing the current plan, image/video credit balances, and placeholder plan cards ready for a future Stripe Checkout integration
- Image download link
- Text-to-video prompt input in the browser
- Optional OpenRouter/Wan generated audio for video clips
- Cost-saving video workflow:
  1. Hermes optimizes the user's prompt for a stable first-frame composition
  2. Build a free 3-frame static storyboard grid with Pollinations: start, middle, and end
  3. Save each storyboard frame as an app-served static image URL so OpenRouter/Wan receives a stable file instead of a dynamic generator URL
  4. Let the user regenerate any individual storyboard frame with a custom prompt before spending video credits
  5. Submit the selected/approved start frame into OpenRouter as `frame_images[0]` / `first_frame`
  6. Animate it with OpenRouter model `alibaba/wan-2.6`
- Video aspect ratio selector:
  - `16:9` widescreen
  - `9:16` vertical
- Async video job polling, video preview, and video download link on the same page
- Video audio toggle that passes `generate_audio=true` to OpenRouter when enabled

## Environment

Video generation requires an OpenRouter API key. OpenAI image edits for masked replacement edits require an OpenAI API key. User login/history requires MongoDB. The free-frame visual review uses your local Hermes Agent CLI, so Hermes must be installed and configured with a vision-capable model/provider.

```bash
cp .env.example .env
# edit .env and set MONGODB_URI and MONGODB_PASSWORD for user accounts/history
# edit .env and set FLASK_SECRET_KEY to a long random string for sessions
# optional: set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET for Google login
# edit .env and set OPENROUTER_API_KEY for video generation
# edit .env and set OPENAI_API_KEY for OpenAI masked image edits
# optional: set FAL_KEY or FAL_API_KEY and INPAINT_PROVIDER=fal for FLUX/Fal.ai inpainting
# verify Hermes is available and configured:
hermes doctor
```

`.env` is ignored by git. Do not commit your real API keys.

Optional variables:

- `MONGODB_URI` - MongoDB Atlas connection URI; can contain `<db_password>` placeholder
- `MONGODB_PASSWORD` - password used to replace `<db_password>` in `MONGODB_URI`
- `MONGODB_DB` - database name; defaults to `tti_app`
- `FLASK_SECRET_KEY` - long random string used to sign login sessions
- `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` - optional Google OAuth credentials for `/auth/google`; add the deployed callback URL `/auth/google/callback` in Google Cloud Console. On Railway, the app honors forwarded HTTPS proxy headers so the callback is generated as `https://.../auth/google/callback`.
- `GOOGLE_REDIRECT_URI` - optional exact Google OAuth callback override, useful if Google still reports `redirect_uri_mismatch`; set it to the exact allowlisted URL such as `https://texttoimage-production-f09d.up.railway.app/auth/google/callback`
- `OPENROUTER_HTTP_REFERER` - your deployed site URL
- `OPENROUTER_APP_TITLE` - app title shown to OpenRouter
- `POLLINATIONS_MODEL` - free image model for browser images and I2V first frames; defaults to `flux` to avoid the public GPT image queue/rate-limit message
- `POLLINATIONS_FALLBACK_MODELS` - comma-separated fallback image models if Pollinations queues/rate-limits the preferred first-frame model; defaults to `turbo,gpt-image-large`
- `POLLINATIONS_TOKEN` - token from https://auth.pollinations.ai for higher Pollinations limits
- `OPENAI_API_KEY` - OpenAI API key for default masked image replacement edits
- `OPENAI_IMAGE_EDIT_MODEL` - OpenAI image edit model; defaults to `gpt-image-1`
- `OPENAI_IMAGE_SIZE` - image size parameter for OpenAI edits; defaults to `auto`
- `OPENAI_IMAGE_QUALITY` - image quality parameter for OpenAI edits; defaults to `auto`
- `INPAINT_PROVIDER` - masked replacement backend; defaults to `openai`, set to `fal` to use Fal.ai/FLUX
- `FAL_KEY` or `FAL_API_KEY` - Fal.ai key for optional FLUX inpainting masked replacement edits
- `FAL_INPAINT_ENDPOINT` - override the FLUX inpainting queue endpoint; defaults to `https://queue.fal.run/fal-ai/flux-general/inpainting`
- `FAL_INPAINT_IMAGE_SIZE` - optional Fal image size hint such as `landscape_16_9`
- `FAL_INPAINT_STEPS` - FLUX inpainting inference steps; default `28`
- `INPAINT_MASK_FEATHER_PX` - pixel blur radius for generated mask edges; default `4`
- `IMAGE_EDIT_ASYNC` - run masked edits in a background job with browser polling; default `true` to avoid long request/proxy timeouts
- `IMAGE_EDIT_JOB_TTL_SECONDS` - seconds to keep completed/failed masked edit jobs in memory; default `3600`
- `PUBLIC_BASE_URL` or `APP_BASE_URL` - optional public HTTPS base URL used for app-served storyboard/mask/source image URLs sent to OpenRouter or Fal.ai; not required for OpenAI image edits because the app uploads image bytes
- `VIDEO_ORCHESTRATOR_REVIEWER=hermes` - uses Hermes Agent as the visual reviewer before paid I2V
- `HERMES_COMMAND` - path/name of the Hermes executable; default `hermes`
- `PROMPT_REWRITE_PROVIDER` and `PROMPT_REWRITE_MODEL` - provider/model override for the Hermes AI prompt rewrite subprocess; defaults are `openrouter` and `openai/gpt-4o-mini`
- `PROMPT_REWRITE_TIMEOUT` - seconds to wait for AI prompt rewriting; default `30`
- `HERMES_REVIEW_PROVIDER` and `HERMES_REVIEW_MODEL` - provider/model override for the Hermes review subprocess; defaults are `openrouter` and `openai/gpt-4o-mini`
- `VIDEO_ORCHESTRATOR_PROMPT_OPTIMIZER=hermes` - optionally asks Hermes to optimize the text prompt too; default `local`
- `VIDEO_ORCHESTRATOR_MAX_ATTEMPTS` - number of free-frame review attempts per request; default `1` for web timeout safety
- `VIDEO_ORCHESTRATOR_REQUIRE_VISION=true` - blocks video generation if the selected visual reviewer is unavailable
- `VIDEO_ORCHESTRATOR_SOFT_REVIEW_FAILURES=true` - default soft gate: continue when Hermes returns unreadable JSON, exits nonzero, times out, or produces a low-confidence rejection
- `VIDEO_ORCHESTRATOR_REJECTION_CONFIDENCE_THRESHOLD=0.85` - low-confidence rejections below this threshold are approved in soft mode
- `VIDEO_ORCHESTRATOR_STRICT_REVIEW=true` - opt back into strict block-on-any-review-failure behavior
- `HERMES_REVIEW_TIMEOUT` - seconds to wait for Hermes visual review; default `30`
- `VIDEO_ORCHESTRATOR_ALLOW_REVIEW_TIMEOUT=true` - if Hermes review times out, proceed using the structurally valid free frame instead of blocking paid I2V

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
- `Database.py` - MongoDB URI/env loading and database connection helper
- `AuthService.py` - user registration/login/password hashing and generation history helpers
- `TexttoImage.py` - Pollinations image generation helper
- `ImageEdit.py` - masked edit helpers for OpenAI/Fal masks, legacy composites, and deterministic recolors
- `OpenAIImageEdit.py` - OpenAI Images Edit client for uploading source image + alpha mask
- `FluxInpaint.py` - optional Fal.ai FLUX inpainting queue client
- `PromptRewrite.py` - Hermes CLI prompt rewrite helper for image/video prompt engineering
- `Storyboard.py` - Pollinations 3-frame start/middle/end storyboard helper
- `OrchestratedVideo.py` - Hermes prompt optimizer, free-frame generation, Vision Agent gate, and retry/blocking logic
- `OpenRouterVideo.py` - OpenRouter Wan 2.6 video generation helper
- `templates/index.html` - Web page template
- `static/styles.css` - Styling
- `static/generated/` - Runtime generated images, ignored by git
- `Procfile` and `railway.json` - deployment start commands
- `test_orchestrated_video.py` - regression tests for the vision-gated I2V workflow
