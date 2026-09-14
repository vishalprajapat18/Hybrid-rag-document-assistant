#!/usr/bin/env bash
# Starts both processes inside one container (Hugging Face Spaces exposes a single port).
# The FastAPI backend runs in the background; Streamlit is the public app on port 7860.
set -e

uvicorn main:app --host 127.0.0.1 --port 8000 &

# Wait until the API has loaded its models and answers /health, so the UI never
# shows "Could not connect to the FastAPI backend" on first load.
for _ in $(seq 1 90); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; then
    break
  fi
  sleep 2
done

exec streamlit run frontend/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-7860}" \
  --server.enableCORS false \
  --server.enableXsrfProtection false
