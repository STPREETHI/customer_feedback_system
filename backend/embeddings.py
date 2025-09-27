import pandas as pd
import numpy as np
import faiss
import os
import bz2
from sentence_transformers import SentenceTransformer

# --- Local Imports ---
# These must be imported correctly for the script to run
from preprocessing import clean_text
from models import sentiment_pipeline
import database

# --- Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'amazon_reviews.csv')
DB_PATH = os.path.join(os.path.dirname(__file__), 'feedback.db')
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'review_index.faiss')
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_REVIEWS_TO_PROCESS = 2000 # Limit for speed

def process_and_store_reviews():
    """
    The main one-time setup function. It reads the raw data, processes it,
    creates embeddings, builds a FAISS index, and stores everything in the DB.
    """
    # --- Check if setup has already been done ---
    if os.path.exists(DB_PATH) and os.path.exists(FAISS_INDEX_PATH):
        print("Database and FAISS index already exist. Skipping processing.")
        print("To re-process, delete 'feedback.db' and 'review_index.faiss' from the 'backend' folder.")
        return

    # --- Step 1: Load and Parse Data ---
    print("--- Starting data processing and setup... ---")
    try:
        # This robust method handles the compressed, non-standard format
        reviews = []
        with bz2.open(DATA_PATH, "rt", encoding="utf-8") as bz_file:
            for i, line in enumerate(bz_file):
                if i >= NUM_REVIEWS_TO_PROCESS:
                    break
                try:
                    # Extract rating and text based on the file's format "__label__X ..."
                    rating_part = line.split(" ")[0]
                    text = line.replace(rating_part, "").strip()
                    rating = int(rating_part.replace("__label__", ""))
                    reviews.append({'rating': rating, 'text': text})
                except Exception:
                    continue # Skip malformed lines
        df = pd.DataFrame(reviews)
        print(f"--- Loaded and parsed {len(df)} reviews from the data file. ---")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_PATH}")
        return
    except Exception as e:
        print(f"ERROR reading data file: {e}")
        return

    # --- Step 2: Clean Text and Analyze Sentiment ---
    print("--- Cleaning text and analyzing sentiment... ---")
    df['cleaned_text'] = df['text'].apply(clean_text)
    # Get sentiment labels (POSITIVE/NEGATIVE)
    sentiments = [res['label'] for res in sentiment_pipeline(df['text'].tolist())]
    df['sentiment'] = sentiments
    print("--- Text cleaning and sentiment analysis complete. ---")

    # --- Step 3: Generate Embeddings ---
    print("--- Generating embeddings... This will take a while. ---")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
    print("--- Embedding generation complete. ---")

    # --- Step 4: Build FAISS Index ---
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index)
    ids = np.array(range(0, len(df))).astype('int64')
    index.add_with_ids(embeddings.astype('float32'), ids)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"--- FAISS index built and saved to {FAISS_INDEX_PATH} ---")

    # --- Step 5: Store in Database ---
    print("--- Storing processed data in the database... ---")
    df['product_id'] = pd.qcut(df.index, q=5, labels=[f"PROD10{i+1}" for i in range(5)])
    
    conn = database.get_db_connection(DB_PATH)
    if conn:
        database.setup_database(conn)
        feedback_to_store = [
            {'review_text': row['text'], 'sentiment': row['sentiment'], 'product_id': row['product_id']}
            for index, row in df.iterrows()
        ]
        database.insert_feedback(conn, feedback_to_store)
        conn.close()
        print("--- Data successfully stored in the database. ---")
    
    print("\n--- ✅ Data processing and setup complete! ---")

if __name__ == '__main__':
    process_and_store_reviews()

