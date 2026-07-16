#!/bin/bash
# Vercel build script for static files
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Copying static files to Vercel output ==="
mkdir -p .vercel/output/static
cp -r static/* .vercel/output/static/
echo "=== Static files copied ==="
ls -la .vercel/output/static/