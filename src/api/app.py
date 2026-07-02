# src/api/app.py
"""FastAPI API for ConvinceSense.
Provides:
- GET /health – simple health check
- GET /config – version/info
- WebSocket /ws/records – real‑time stream of Record objects
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from src.pipelines.live_pipeline import ConvinceSensePipeline
from src.features.engagement.tracker import EngagementRecord
import json
import os
import queue
import asyncio

app = FastAPI()

# CORS – allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional simple API key authentication (environment variable)
API_KEY = os.getenv("CONVINCESENSE_API_KEY")

def _require_api_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Initialize the live pipeline (do not start it automatically on server start)
pipeline = ConvinceSensePipeline()

@app.get("/health")
async def health(x_api_key: str = Header(None)):
    _require_api_key(x_api_key)
    return {"status": "ok"}

@app.get("/config")
async def config(x_api_key: str = Header(None)):
    _require_api_key(x_api_key)
    return {"version": "1.0.0", "framework": "ConvinceSense"}

@app.get("/session/summary")
async def session_summary(x_api_key: str = Header(None)):
    _require_api_key(x_api_key)
    try:
        summary_text = pipeline.get_summary()
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/records")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    # Reset tracker and clear any left-over records in the output queue
    pipeline.tracker.reset()
    while not pipeline.output_q.empty():
        try:
            pipeline.output_q.get_nowait()
        except queue.Empty:
            break
            
    # Start the live pipeline on connection
    pipeline.start()
    print("[API WS] Client connected. Started live pipeline.")
    
    # Task to read messages from client to detect disconnects
    async def receive_messages():
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    # Task to stream records to client
    async def send_records():
        loop = asyncio.get_running_loop()
        try:
            while True:
                # Fetch next record from queue (blocking call offloaded to thread executor)
                record = await loop.run_in_executor(None, lambda: pipeline.output_q.get())
                await ws.send_text(json.dumps(record.__dict__))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[API WS] Error sending records: {e}")

    receive_task = asyncio.create_task(receive_messages())
    send_task = asyncio.create_task(send_records())
    
    try:
        await asyncio.wait(
            [receive_task, send_task],
            return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        receive_task.cancel()
        send_task.cancel()
        pipeline.stop()
        print("[API WS] Client disconnected. Stopped live pipeline.")
