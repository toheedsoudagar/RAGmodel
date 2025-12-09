#!/usr/bin/env bash
set -euo pipefail

# This entrypoint runs basic DB setup and ingestion if present, then starts the default process.
# Note: When used in the "app" service, it will launch Streamlit. The backend service overrides command to run uvicorn.

echo "[entrypoint] Starting entrypoint in $(pwd) at $(date)"

# Run DB setup if exists (safe to fail)
if [ -f ./db_setup.py ]; then
  echo "[entrypoint] Running db_setup.py..."
  python db_setup.py || echo "[entrypoint] db_setup.py completed with non-zero exit (continuing)."
else
  echo "[entrypoint] No db_setup.py found, skipping."
fi

# Run ingestion (safe to fail)
if [ -f ./ingest.py ]; then
  echo "[entrypoint] Running ingest.py..."
  python ingest.py || echo "[entrypoint] ingest.py completed with non-zero exit (continuing)."
else
  echo "[entrypoint] No ingest.py found, skipping."
fi

# If running as app container (Streamlit), start the UI; otherwise container command overrides
if [ -f ./streamlit_frontend.py ] || [ -f ./app.py ]; then
  ENTRY_STREAMLIT=${STREAMLIT_ENTRY:-streamlit_frontend.py}
  if [ -f "$ENTRY_STREAMLIT" ]; then
    echo "[entrypoint] Starting Streamlit: $ENTRY_STREAMLIT"
    streamlit run "$ENTRY_STREAMLIT" --server.port=8501 --server.address=0.0.0.0
  else
    echo "[entrypoint] Streamlit entrypoint $ENTRY_STREAMLIT not found. Sleeping..."
    tail -f /dev/null
  fi
else
  echo "[entrypoint] No Streamlit file found. Sleeping..."
  tail -f /dev/null
fi
