import sqlite3
import os

DB_NAME = "nurvek_detections.db"

def check_database():
    if not os.path.exists(DB_NAME):
        print(f"Database file '{DB_NAME}' does not exist.")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detected_records';")
        if cursor.fetchone() is None:
            print(f"Table 'detected_records' does not exist in '{DB_NAME}'.")
            conn.close()
            return

        # Get total record count
        cursor.execute("SELECT COUNT(*) FROM detected_records")
        count = cursor.fetchone()[0]
        print(f"Total records in 'detected_records': {count}")

        if count > 0:
            print("\nFirst 5 records (or fewer if less than 5 exist):")
            # Fetch column names to make output more readable
            cursor.execute("PRAGMA table_info(detected_records);")
            columns = [info[1] for info in cursor.fetchall()]
            print(f"Columns: {columns}")

            cursor.execute("SELECT * FROM detected_records LIMIT 5")
            for i, row in enumerate(cursor.fetchall()):
                print(f"Record {i+1}: {row}")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_database()