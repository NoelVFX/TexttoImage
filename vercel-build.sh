#!/bin/bash
# Vercel build script for static files
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Copying static files to public folder ==="
mkdir -p public
cp -r static/* public/
echo "=== Public files copied ==="
ls -la public/