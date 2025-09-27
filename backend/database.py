import sqlite3
import os

def get_db_connection(db_path):
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def setup_database(conn):
    """Creates the feedback table if it doesn't exist."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            sentiment TEXT,
            product_id TEXT
        );
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database setup error: {e}")

def insert_feedback(conn, feedback_data):
    """Inserts a list of feedback data into the database."""
    try:
        cursor = conn.cursor()
        cursor.executemany("""
        INSERT INTO feedback (review_text, sentiment, product_id)
        VALUES (:review_text, :sentiment, :product_id);
        """, feedback_data)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database insert error: {e}")

