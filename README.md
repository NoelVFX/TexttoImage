# Text to Image Web App

A small Flask web app that turns a text prompt into an image using Pollinations and displays the result on the same page with a download button.

## Features

- Prompt input in the browser
- Aspect ratio selector:
  - `1024x1024` / 1:1 square
  - `1792x1024` / 16:9 widescreen
- Generated image preview on the same page
- Download link for each generated image

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

`gunicorn` is already included in `requirements.txt` and `Procfile` is included for Heroku-compatible platforms.

## Push to GitHub

```bash
git init -b main
git add app.py TexttoImage.py templates/index.html static/styles.css static/generated/.gitkeep requirements.txt Procfile README.md .gitignore
git commit -m "Build Flask text-to-image web app"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## Project files

- `app.py` - Flask routes and web UI integration
- `TexttoImage.py` - Pollinations image generation helper
- `templates/index.html` - Web page template
- `static/styles.css` - Styling
- `static/generated/` - Runtime generated images, ignored by git
