# app/routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.ws_manager import manager
import asyncio

router = APIRouter()


@router.websocket("/stream")
async def stream_events(websocket: WebSocket):
    """
    WebSocket endpoint for live event streaming.
    Clients connect here to receive real-time road event notifications.
    The server sends a keepalive ping every 30 seconds.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Send a keepalive ping every 30 seconds
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
