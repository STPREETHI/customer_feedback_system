import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'feedback.db')

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = None
    try:
        # check_same_thread=False is needed for Flask's multi-threaded environment
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # Allows accessing columns by name (e.g., row['product_id'])
        conn.row_factory = sqlite3.Row 
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
    return conn

def setup_database():
    """Creates the feedback table if it doesn't already exist."""
    conn = get_db_connection()
    if conn is None:
        print("CRITICAL: Could not create database connection. Table setup failed.")
        return
    try:
        cursor = conn.cursor()
        # Drop the table if it exists to ensure a fresh start during setup
        cursor.execute("DROP TABLE IF EXISTS feedback")
        cursor.execute('''
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY,
                product_id TEXT NOT NULL,
                review_text TEXT NOT NULL,
                cleaned_text TEXT,
                sentiment TEXT,
                sentiment_score REAL
            )
        ''')
        conn.commit()
        print("Database table 'feedback' created successfully.")
    except sqlite3.Error as e:
        print(f"Database setup error: {e}")
    finally:
        if conn:
            conn.close()

def insert_feedback_batch(feedback_data_list):
    """Inserts a batch of review records into the feedback table."""
    conn = get_db_connection()
    if conn is None:
        print("CRITICAL: Could not create database connection. Insertion failed.")
        return
        
    sql = ''' INSERT INTO feedback(id, product_id, review_text, cleaned_text, sentiment, sentiment_score)
              VALUES(?,?,?,?,?,?) '''
    try:
        cursor = conn.cursor()
        cursor.executemany(sql, feedback_data_list)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Failed to insert feedback batch: {e}")
    finally:
        if conn:
            conn.close()

