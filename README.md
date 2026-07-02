# Text to Image and Video Web App

A Flask web app that turns prompts into images with Pollinations and videos with OpenRouter's ByteDance Seedance 2.0 Fast model.

## Features

- Text-to-image prompt input in the browser
- Image aspect ratio selector:
  - `1024x1024` / 1:1 square
  - `1792x1024` / 16:9 widescreen
- Generated image preview on the same page
- Image download link
- Text-to-video prompt input in the browser
- Video generation through OpenRouter model `bytedance/seedance-2.0-fast`
- Video aspect ratio selector:
  - `1:1` square
  - `16:9` widescreen
- Async video job polling, video preview, and video download link on the same page

## Environment

Video generation requires an OpenRouter API key.

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
```

`.env` is ignored by git. Do not commit your real API key.

Optional variables:

- `OPENROUTER_HTTP_REFERER` - your deployed site URL
- `OPENROUTER_APP_TITLE` - app title shown to OpenRouter

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000.

## Deploy

This is a standard Flask app. For platforms like Render, Railway, Fly.io, or Heroku-style hosts:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Environment variable: `OPENROUTER_API_KEY`

`gunicorn` is already included in `requirements.txt` and `Procfile` is included for Heroku-compatible platforms.

## Push to GitHub

```bash
git init -b main
git add app.py TexttoImage.py OpenRouterVideo.py templates/index.html static/styles.css static/generated/.gitkeep requirements.txt Procfile README.md .gitignore .env.example
git commit -m "Build Flask text-to-image and video web app"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## Project files

- `app.py` - Flask routes and web UI/API integration
- `TexttoImage.py` - Pollinations image generation helper
- `OpenRouterVideo.py` - OpenRouter Seedance 2.0 Fast video generation helper
- `templates/index.html` - Web page template
- `static/styles.css` - Styling
- `static/generated/` - Runtime generated images, ignored by git
