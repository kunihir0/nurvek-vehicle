import sqlite3
from typing import Any, Tuple, List

# Type hint for database connection and cursor
DbConnection = sqlite3.Connection
DbCursor = sqlite3.Cursor

def init_db_connection(db_name: str) -> Tuple[DbConnection, DbCursor]:
    """Initializes the database connection and creates the table if it doesn't exist."""
    conn = sqlite3.connect(db_name, check_same_thread=False) 
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detected_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, frame_number INTEGER,
            track_id INTEGER, vehicle_class TEXT, vehicle_confidence REAL,
            vehicle_bbox TEXT, lp_confidence REAL, lp_bbox_local TEXT, lp_text TEXT 
        )
    ''')
    conn.commit()
    print(f"[DB] Database '{db_name}' connection opened and table ensured.")
    return conn, cursor

def flush_db_batch(conn: DbConnection, cursor: DbCursor, records_batch: List[Tuple[Any, ...]]) -> None:
    """Flushes a batch of records to the database."""
    if not records_batch:
        return
    try:
        cursor.executemany('''
            INSERT INTO detected_records (timestamp, frame_number, track_id, vehicle_class, 
            vehicle_confidence, vehicle_bbox, lp_confidence, lp_bbox_local, lp_text) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', records_batch)
        conn.commit()
    except sqlite3.Error as e:
        print(f"[DB_ERROR] Failed to flush batch: {e}")
    records_batch.clear()