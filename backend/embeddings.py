import os
import bz2
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

# Import from other backend modules
from preprocessing import clean_text
from models import predict_sentiment
from database import setup_database, insert_feedback_batch, get_db_connection

# --- Constants and Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'amazon_reviews.csv')
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'review_index.faiss')
DB_PATH = os.path.join(os.path.dirname(__file__), 'feedback.db')
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLE_SIZE = 5000 # Limit the number of reviews to process for a faster initial setup

# --- Global Variables ---
embedding_model = None
faiss_index = None
df_reviews_indexed = None

def load_embedding_model():
    """Loads the sentence transformer model into memory."""
    global embedding_model
    if embedding_model is None:
        print(f"Loading embedding model ({EMBEDDING_MODEL_NAME}) onto {DEVICE}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        print("Embedding model loaded successfully.")

def get_faiss_index_and_data():
    """Loads the FAISS index and corresponding review data into memory."""
    global faiss_index, df_reviews_indexed
    if faiss_index is None:
        print("Loading FAISS index and review data...")
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Please run `python backend/embeddings.py` first.")
        
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        conn = get_db_connection()
        if conn is None:
             raise ConnectionError("Failed to connect to the database to load indexed data.")
        df_reviews_indexed = pd.read_sql_query("SELECT * FROM feedback", conn)
        conn.close()
        
        # Set the original DB id as the dataframe index for fast .loc lookups
        if 'id' in df_reviews_indexed.columns:
            df_reviews_indexed.set_index('id', inplace=True)
        print("FAISS index and data loaded.")
    return faiss_index, df_reviews_indexed

def find_similar_reviews(query_text, k=5):
    """Finds the k most similar reviews to a query text using FAISS."""
    index, data = get_faiss_index_and_data()
    if index is None or data is None:
        return pd.DataFrame()

    load_embedding_model() # Ensure model is loaded
    query_embedding = embedding_model.encode([query_text], device=DEVICE)
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    # Filter out invalid indices which FAISS returns as -1
    valid_indices = [i for i in indices[0] if i != -1]
    
    # Retrieve reviews using the dataframe's index (.loc is efficient for this)
    similar_reviews = data.loc[valid_indices]
    return similar_reviews

def full_data_processing_pipeline():
    """
    The main one-time script to process the raw data and build the database and FAISS index.
    """
    print("--- Starting Full Data Processing Pipeline ---")
    
    # 1. Load and Parse Raw Data
    print(f"Loading and parsing raw data from {DATA_PATH}...")
    try:
        data = []
        with bz2.open(DATA_PATH, "rt", encoding="utf-8") as bz2_file:
            for i, line in enumerate(bz2_file):
                if i >= SAMPLE_SIZE: # Limit sample size
                    break
                try:
                    # Format is __label__X ...text...
                    # We are not using the original rating, sentiment model will provide it
                    text = line[11:].strip()
                    if text: # Ensure text is not empty
                        data.append(text)
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data, columns=['review_text'])
        df.reset_index(inplace=True) # Use default integer index as a unique ID
        df.rename(columns={'index': 'id'}, inplace=True)
        print(f"Loaded and parsed {len(df)} reviews.")
    except Exception as e:
        print(f"FATAL: Error reading or processing raw file: {e}")
        return

    # 2. Clean Text
    print("Cleaning review text...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    df.dropna(subset=['cleaned_text'], inplace=True)

    # 3. Predict Sentiment
    print("Predicting sentiment for reviews...")
    sentiment_results = predict_sentiment(df['cleaned_text'].tolist())
    df['sentiment'] = [res['label'] for res in sentiment_results]
    # Standardize score: POSITIVE is score, NEGATIVE is 1 - score (closer to 1 is more negative)
    df['sentiment_score'] = [res['score'] if res['label'] == 'POSITIVE' else 1 - res['score'] for res in sentiment_results]

    # 4. Assign Dummy Product IDs for demonstration
    df['product_id'] = [f'PROD{101 + (i % 10)}' for i in range(len(df))]

    # 5. Generate Embeddings
    print("Generating embeddings for reviews...")
    load_embedding_model()
    embeddings = embedding_model.encode(df['cleaned_text'].tolist(), show_progress_bar=True, device=DEVICE)

    # 6. Build and Save FAISS Index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index) # Map vectors to original dataframe IDs
    index.add_with_ids(np.array(embeddings).astype('float32'), df['id'].values.astype('int64'))
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index built and saved to {FAISS_INDEX_PATH}")

    # 7. Store in Database
    print("Storing processed data in SQLite database...")
    setup_database() # Create a fresh table
    
    db_records = df[['id', 'product_id', 'review_text', 'cleaned_text', 'sentiment', 'sentiment_score']].to_records(index=False)
    insert_feedback_batch(list(db_records))
    
    print("--- Data Processing Pipeline Finished Successfully ---")

if __name__ == "__main__":
    # This allows running `python backend/embeddings.py` directly to perform the initial setup
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DB_PATH):
        print("Database and FAISS index already exist. Skipping processing.")
        print("To re-process, delete 'feedback.db' and 'review_index.faiss' from the 'backend' folder.")
    else:
        full_data_processing_pipeline()
