# app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ingest, events, cameras, heatmap, websocket, media, assets
from app.config import settings

app = FastAPI(
    title="Smart City Road Intelligence API",
    version="1.0.0",
    description="Real-time road event detection and analytics platform — YOLO26 + Road Infrastructure Measurement Edition",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for debugging on physical devices
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"\n>>> [Request] {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"<<< [Response] {response.status_code}")
    return response

app.include_router(ingest.router,    prefix="/ingest",   tags=["Ingest"])
app.include_router(events.router,    prefix="/events",   tags=["Events"])
app.include_router(cameras.router,   prefix="/cameras",  tags=["Cameras"])
app.include_router(heatmap.router,   prefix="/heatmap",  tags=["Heatmap"])
app.include_router(assets.router,    prefix="/assets",    tags=["Assets"])
app.include_router(websocket.router, prefix="/ws",       tags=["Stream"])
app.include_router(media.router,     prefix="/media",    tags=["Media"])

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Smart City Road Intelligence API",
        "version": "1.0.0",
        "status": "running",
    }

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}

@app.get("/debug/headers", tags=["Health"])
async def debug_headers(request: Request):
    print(f"\n[debug] Request from IP: {request.client.host}")
    print(f"[debug] Headers: {dict(request.headers)}")
    return {"ip": request.client.host, "headers": dict(request.headers)}
