import fastapi
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse # StreamingResponse not directly used for SSE here
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse # Moved import to top
import pathlib
import datetime # Added for LiveFrameData timestamp
import asyncio # For SSE
import json # For SSE data formatting

from pydantic import BaseModel, Field
import uvicorn
from typing import Dict, Any, List, Optional

from src.core.schemas import VehicleEvent
from src.config import settings
from src.core.qdrant_sync import initialize_qdrant_resources
# Import the globals directly to be used after initialization
from src.core.qdrant_sync import qdrant_client as qdrant_sync_client
from src.core.qdrant_sync import embedding_model as qdrant_sync_embedding_model


# --- Global variable to store the latest live frame ---
latest_live_frame_base64: Optional[str] = None
# ---

# --- SSE OCR Stream Globals (REMOVED as EasyOCR is now used directly in backend) ---
# ocr_stream_client_queues: List[asyncio.Queue] = []
backend_status_client_queues: List[asyncio.Queue] = [] # For general backend status
# ---

app = FastAPI(
    title="Nurvek Vehicle Arm API",
    description="API for receiving vehicle detection and attribute events, serving live feed, backend status streams, and semantic search.",
    version="0.1.4" # Incremented version
)

@app.on_event("startup")
async def startup_event():
    print("[API_SERVER] Initializing Qdrant resources on startup...")
    try:
        initialize_qdrant_resources()
        print("[API_SERVER] Qdrant resources initialized successfully.")
    except Exception as e:
        print(f"[API_SERVER_ERROR] Failed to initialize Qdrant resources on startup: {e}")
        # Depending on severity, you might want to prevent app startup or handle gracefully
        # For now, endpoints relying on Qdrant will fail if this doesn't succeed.

# --- CORS Middleware Setup ---
origins = [
    "http://localhost:1242", # Default for local dev
    "http://127.0.0.1:1242",
    "http://0.0.0.0:1242", # If accessing from other devices on network via 0.0.0.0
    # Add other origins if needed, e.g., specific frontend dev server ports
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local dev, more permissive for SSE
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Static Files Setup ---
current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent # src -> nurvek-vehicle
web_dir = project_root / "web"

# Serve index.html at root
@app.get("/", include_in_schema=False)
async def get_index_html():
    index_path = web_dir / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

# Serve static assets (CSS, JS)
app.mount("/css", StaticFiles(directory=web_dir / "css"), name="css")
app.mount("/js", StaticFiles(directory=web_dir / "js"), name="js")


# --- Vehicle Event Logging ---
received_events_log: list[VehicleEvent] = []

@app.post("/api/v1/vehicle_event", response_model=Dict[str, Any])
async def post_vehicle_event(event: VehicleEvent):
    """
    Receives vehicle event data from the Nurvek Vehicle Arm.
    """
    print(f"[API_SERVER] Received Vehicle Event for Track ID: {event.vehicle_track_id} from Camera: {event.camera_id}")
    received_events_log.append(event)
    return {"status": "success", "message": "Vehicle event received", "track_id": event.vehicle_track_id}

@app.get("/api/v1/vehicle_events/log", response_model=List[VehicleEvent])
async def get_received_events_log():
    """
    Returns a log of all vehicle events received by this API instance.
    """
    return received_events_log

@app.get("/api/v1/track_details/{track_id}", response_model=List[VehicleEvent])
async def get_track_id_details(track_id: int):
    """
    Returns all recorded event details for a specific vehicle_track_id.
    """
    print(f"[API_SERVER] Request for track_id details: {track_id}")
    matching_events = [event for event in received_events_log if event.vehicle_track_id == track_id]
    if not matching_events:
        # It's better to return an empty list if not found, rather than 404,
        # as the track ID might be valid but have no *logged* events yet, or events were cleared.
        # The client-side JS tool can then inform Gemma if no data was found.
        print(f"[API_SERVER] No events found for track_id: {track_id}")
        return []
    return matching_events

# --- Live Frame Endpoints ---
class DetectionBoxData(BaseModel):
    track_id: Optional[int] = None
    class_name: str
    confidence: float
    box_coords: List[int] 

class LiveFrameData(BaseModel):
    frame_base64: str
    detections: List[DetectionBoxData] = []
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

latest_frame_and_detections: Optional[LiveFrameData] = None

@app.post("/api/v1/internal/update_live_frame", include_in_schema=False)
async def update_live_frame(frame_data: LiveFrameData):
    global latest_frame_and_detections
    latest_frame_and_detections = frame_data
    return {"status": "success", "message": "Live frame and detections updated"}

@app.get("/api/v1/live_feed_frame", response_model=Optional[LiveFrameData])
async def get_live_feed_frame():
    """
    Returns the latest live frame as base64 encoded data, along with detections.
    """
    return latest_frame_and_detections

# --- OCR Stream Endpoints (REMOVED as EasyOCR is now used directly in backend) ---
# class OcrChunkData(BaseModel): ...
# @app.post("/api/v1/internal/ocr_stream_chunk", include_in_schema=False) ...
# async def ocr_event_generator(request: Request, client_queue: asyncio.Queue): ...
# @app.get("/api/v1/ocr_stream_feed", include_in_schema=False) ...

# --- Backend Status Stream Endpoints ---
class BackendStatusUpdate(BaseModel):
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    source: str # e.g., "pipeline_main_loop", "ocr_worker"
    event_type: str # e.g., "status", "image_processed", "error"
    data: Dict[str, Any] # Flexible data payload

@app.post("/api/v1/internal/backend_status_update", include_in_schema=False)
async def post_backend_status_update(status_update: BackendStatusUpdate):
    # print(f"[API_SERVER_STATUS_UPDATE] Received status: {status_update.source} - {status_update.event_type}")
    for q in backend_status_client_queues:
        await q.put(status_update.model_dump_json())
    return {"status": "success", "message": "Backend status update received"}

async def backend_status_event_generator(request: Request, client_queue: asyncio.Queue):
    try:
        while True:
            if await request.is_disconnected():
                print("[API_SERVER_SSE] Backend Status client disconnected.")
                break
            try:
                message = await asyncio.wait_for(client_queue.get(), timeout=1.0)
                yield f"event: backend_status\ndata: {message}\n\n"
            except asyncio.TimeoutError:
                yield ": keep-alive-status\n\n"
                continue
    except asyncio.CancelledError:
        print("[API_SERVER_SSE] Backend Status generator cancelled.")
    finally:
        if client_queue in backend_status_client_queues:
            backend_status_client_queues.remove(client_queue)
        print(f"[API_SERVER_SSE] Backend Status client queue removed. Remaining: {len(backend_status_client_queues)}")

@app.get("/api/v1/backend_status_feed", include_in_schema=False)
async def backend_status_feed(request: Request):
    client_queue = asyncio.Queue()
    backend_status_client_queues.append(client_queue)
    print(f"[API_SERVER_SSE] New Backend Status client connected. Total: {len(backend_status_client_queues)}")
    return EventSourceResponse(backend_status_event_generator(request, client_queue), ping=15)

# --- Semantic Search Endpoint ---
class SemanticSearchQuery(BaseModel):
    query_text: str
    top_k: Optional[int] = Field(default=5, ge=1, le=50)

class SemanticSearchResultItem(BaseModel):
    id: str # Qdrant point ID
    score: float
    payload: Dict[str, Any] # The payload stored in Qdrant

class SemanticSearchResults(BaseModel):
    results: List[SemanticSearchResultItem]
    query_text: str
    count: int

@app.post("/api/v1/events/semantic_search", response_model=SemanticSearchResults) # Changed to POST to accept body
async def semantic_search_events(search_query: SemanticSearchQuery):
    """
    Performs semantic search over stored vehicle events using Qdrant.
    """
    if qdrant_sync_client is None or qdrant_sync_embedding_model is None:
        print("[API_SERVER_ERROR] Qdrant client or embedding model not available for semantic search.")
        raise HTTPException(status_code=503, detail="Semantic search service not available. Resources not initialized.")

    try:
        print(f"[API_SEMANTIC_SEARCH] Received query: '{search_query.query_text}', top_k: {search_query.top_k}")
        query_vector = qdrant_sync_embedding_model.encode(search_query.query_text).tolist()

        qdrant_search_results = qdrant_sync_client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=search_query.top_k,
            with_payload=True  # Retrieve the payload
        )
        
        formatted_results: List[SemanticSearchResultItem] = []
        for hit in qdrant_search_results:
            formatted_results.append(
                SemanticSearchResultItem(
                    id=str(hit.id), # Ensure ID is string
                    score=hit.score,
                    payload=hit.payload if hit.payload else {}
                )
            )
        
        print(f"[API_SEMANTIC_SEARCH] Found {len(formatted_results)} results for query: '{search_query.query_text}'")
        return SemanticSearchResults(
            results=formatted_results,
            query_text=search_query.query_text,
            count=len(formatted_results)
        )

    except Exception as e:
        print(f"[API_SEMANTIC_SEARCH_ERROR] Error during semantic search for query '{search_query.query_text}': {e}")
        raise HTTPException(status_code=500, detail=f"Error performing semantic search: {str(e)}")


# --- Health Check ---
@app.get("/health", include_in_schema=False)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

# --- Main execution for direct run ---
if __name__ == "__main__":
    print("Starting Uvicorn server for Nurvek API on http://localhost:1242")
    # Consider adding reload=True for development, but remove for "production"
    uvicorn.run(app, host="0.0.0.0", port=1242, log_level="info")