import datetime
from typing import Optional, Dict, Any, List
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.core.schemas import VehicleEvent # Assuming VehicleEvent is the object we'll process

# --- Globals for Qdrant Client and Embedding Model ---
# These will be initialized once when the module is first imported by a process that uses it (e.g., pipeline worker)
qdrant_client: Optional[QdrantClient] = None
embedding_model: Optional[SentenceTransformer] = None
embedding_size: Optional[int] = None

def eprint(*args, **kwargs):
    """Helper to print to stderr for server-side logs."""
    import sys
    print(*args, file=sys.stderr, **kwargs)

def initialize_qdrant_resources():
    """Initializes Qdrant client, embedding model, and ensures collection exists."""
    global qdrant_client, embedding_model, embedding_size

    if qdrant_client is not None and embedding_model is not None:
        return # Already initialized

    eprint("[QDRANT_SYNC] Initializing Qdrant resources...")
    try:
        qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        eprint(f"[QDRANT_SYNC] Qdrant client initialized for host: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

        # Check if collection exists
        try:
            collection_info = qdrant_client.get_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
            eprint(f"[QDRANT_SYNC] Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' already exists.")
            # We need the vector size from the existing collection if not creating
            # Assuming vectors config is present and has size
            if collection_info.config.params.vectors: # type: ignore
                 # If it's a single vector config
                if isinstance(collection_info.config.params.vectors, models.VectorParams): # type: ignore
                    embedding_size = collection_info.config.params.vectors.size # type: ignore
                # If it's a dict of named vectors (less likely for our simple case)
                elif isinstance(collection_info.config.params.vectors, dict) and 'default' in collection_info.config.params.vectors: # type: ignore
                    embedding_size = collection_info.config.params.vectors['default'].size # type: ignore
                else: # Fallback or error if structure is unexpected
                    eprint(f"[QDRANT_SYNC_WARN] Could not determine embedding size from existing collection config. Will rely on model.")


        except Exception as e: # Catching generic exception as qdrant_client might raise different errors
            if "404" in str(e) or "not found" in str(e).lower(): # Heuristic for collection not found
                eprint(f"[QDRANT_SYNC] Collection '{settings.QDRANT_COLLECTION_NAME}' not found. Attempting to create.")
                # Initialize embedding model first to get embedding_size for collection creation
                if embedding_model is None:
                    eprint(f"[QDRANT_SYNC] Initializing sentence transformer model: {settings.EMBEDDING_MODEL_NAME}")
                    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                    # Get embedding dimension
                    dummy_embedding = embedding_model.encode("test")
                    embedding_size = len(dummy_embedding) # type: ignore
                    eprint(f"[QDRANT_SYNC] Embedding model loaded. Vector size: {embedding_size}")

                if embedding_size is None:
                    raise ValueError("Could not determine embedding size for new Qdrant collection.")

                qdrant_client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=embedding_size, distance=models.Distance.COSINE)
                )
                eprint(f"[QDRANT_SYNC] Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' created with vector size {embedding_size}.")
            else:
                eprint(f"[QDRANT_SYNC_ERROR] Error checking/creating Qdrant collection: {e}")
                raise

        # Ensure embedding model is loaded if not done during collection creation
        if embedding_model is None:
            eprint(f"[QDRANT_SYNC] Initializing sentence transformer model: {settings.EMBEDDING_MODEL_NAME}")
            embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            if embedding_size is None: # If not determined from existing collection
                dummy_embedding = embedding_model.encode("test")
                embedding_size = len(dummy_embedding) # type: ignore
            eprint(f"[QDRANT_SYNC] Embedding model loaded. Vector size: {embedding_size}")
        
        eprint("[QDRANT_SYNC] Qdrant resources initialization complete.")

    except Exception as e:
        eprint(f"[QDRANT_SYNC_FATAL] Failed to initialize Qdrant resources: {e}")
        qdrant_client = None # Ensure it's None if init failed
        embedding_model = None
        raise

def create_text_representation_for_event(event: VehicleEvent) -> str:
    """Creates a single text string from VehicleEvent for embedding."""
    parts = []
    parts.append(f"Event at {event.timestamp_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    parts.append(f"on camera {event.camera_id}.")
    parts.append(f"A {event.base_class} (track ID {event.vehicle_track_id}) was detected")
    if event.attributes.license_plate and event.attributes.license_plate.detected and event.attributes.license_plate.text:
        parts.append(f"with license plate {event.attributes.license_plate.text}.")
    else:
        parts.append("with no clear license plate detected.")
    # Could add color if available from a future vision model step for attributes
    return " ".join(parts)

def generate_event_qdrant_id(event: VehicleEvent) -> str:
    """Generates a unique ID for Qdrant point based on event data."""
    # Using timestamp and track_id should be fairly unique.
    # Convert timestamp to a string format suitable for IDs.
    ts_str = event.timestamp_utc.isoformat().replace(":", "-").replace(".", "-")
    return f"event_{ts_str}_{event.vehicle_track_id}"


def embed_and_store_event(event: VehicleEvent, sqlite_event_id: Optional[Any] = None):
    """
    Generates embedding for the event and upserts it to Qdrant.
    'sqlite_event_id' is the ID from the primary SQLite table for this event.
    If no simple ID, this could be a composite key string or a UUID generated when event is created.
    For now, we'll use a generated Qdrant ID and store key event fields in payload.
    """
    global qdrant_client, embedding_model
    if qdrant_client is None or embedding_model is None:
        eprint("[QDRANT_SYNC_ERROR] Qdrant client or embedding model not initialized. Call initialize_qdrant_resources() first.")
        initialize_qdrant_resources() # Attempt to initialize if not already
        if qdrant_client is None or embedding_model is None: # Check again
             eprint("[QDRANT_SYNC_FATAL] Initialization failed. Cannot store event.")
             return

    text_to_embed = create_text_representation_for_event(event)
    vector = embedding_model.encode(text_to_embed).tolist() # type: ignore

    # Payload will store key filterable/retrievable data.
    # The full VehicleEvent could be stored if needed, but can be large.
    # Storing IDs/key fields is often enough if SQLite is the source of truth.
    payload = {
        "timestamp_utc": event.timestamp_utc.isoformat(),
        "camera_id": event.camera_id,
        "vehicle_track_id": event.vehicle_track_id,
        "base_class": event.base_class,
        "lp_text": event.attributes.license_plate.text if event.attributes.license_plate else None,
        "text_summary_for_search": text_to_embed # Store the text that was embedded
    }
    if sqlite_event_id: # If we have a direct ID from SQLite for this specific log entry
        payload["sqlite_id"] = str(sqlite_event_id) # Ensure it's a Qdrant-compatible type

    # Generate a unique ID for the Qdrant point.
    # This could be a UUID generated when VehicleEvent is created, or derived.
    # For the hybrid approach, if SQLite has an auto-inc ID, that's best.
    # If not, a composite string or UUID.
    qdrant_point_id = generate_event_qdrant_id(event)


    try:
        qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=qdrant_point_id, # Use a generated unique ID
                    vector=vector,
                    payload=payload
                )
            ]
        )
        eprint(f"[QDRANT_SYNC] Event upserted to Qdrant. ID: {qdrant_point_id}, TrackID: {event.vehicle_track_id}")
    except Exception as e:
        eprint(f"[QDRANT_SYNC_ERROR] Failed to upsert event to Qdrant (ID: {qdrant_point_id}): {e}")

# Example of how to initialize (e.g., in main.py or at the start of a worker process)
# if __name__ == '__main__':
#     try:
#         initialize_qdrant_resources()
#         # Example usage:
#         # from src.core.schemas import VehicleAttributes, LicensePlateData
#         # test_event_data = VehicleEvent(
#         #     timestamp_utc=datetime.datetime.utcnow(),
#         #     camera_id="CAM_TEST_01",
#         #     vehicle_track_id=12345,
#         #     base_class="car",
#         #     confidence_base_class=0.9,
#         #     bounding_box_frame_coords=[10,10,100,100],
#         #     attributes=VehicleAttributes(
#         #         license_plate=LicensePlateData(text="TESTLP1", detected=True)
#         #     )
#         # )
#         # embed_and_store_event(test_event_data)
#         # eprint("Test event processed.")
#     except Exception as e:
#         eprint(f"Error in Qdrant sync example: {e}")