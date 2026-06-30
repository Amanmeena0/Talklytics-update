# src/api/app.py
"""FastAPI API for ConvinceSense.
Provides:
- GET /health – simple health check
- GET /config – version/info
- WebSocket /ws/records – real‑time stream of Record objects
"""

from fastapi import FastAPI, WebSocket, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from src.pipelines.live_pipeline import ConvinceSensePipeline
from src.features.engagement.tracker import EngagementRecord
import json
import os

app = FastAPI()

# CORS – allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional simple API key authentication (environment variable)
API_KEY = os.getenv("CONVINCESENSE_API_KEY")

def _require_api_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Start the live pipeline once when the app starts
pipeline = ConvinceSensePipeline()
pipeline.start()

@app.get("/health")
async def health(x_api_key: str = Header(None)):
    _require_api_key(x_api_key)
    return {"status": "ok"}

@app.get("/config")
async def config(x_api_key: str = Header(None)):
    _require_api_key(x_api_key)
    return {"version": "1.0.0", "framework": "ConvinceSense"}

@app.websocket("/ws/records")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # No auth for websockets – optionally verify a query param later
    while True:
        # Blocking call to get the next record from the pipeline
        record: EngagementRecord = pipeline.output_q.get()
        await ws.send_text(json.dumps(record.__dict__))
