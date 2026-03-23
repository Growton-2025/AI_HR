#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting Deployment Script..."

# 1. Build the Frontend
echo "📦 Building Frontend..."
cd frontend
npm install
npm run build
cd ..

# 2. Start the Backend (which serves the Frontend)
echo "🐍 Starting Backend..."
cd backend
# Use gunicorn for production stability, or python api.py for simplicity
# Azure App Service expects the app to listen on port 8000 (or $PORT)
./venv/bin/python -m app.main
