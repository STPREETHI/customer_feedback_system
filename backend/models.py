from transformers import pipeline
import torch

# --- Model Configuration ---
SENTIMENT_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Global Model Variable ---
sentiment_pipeline = None

def load_sentiment_model():
    """Loads the sentiment analysis pipeline into memory, if not already loaded."""
    global sentiment_pipeline
    if sentiment_pipeline is None:
        print(f"Loading sentiment analysis model ({SENTIMENT_MODEL_NAME}) onto {DEVICE}...")
        try:
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODEL_NAME,
                framework="pt",
                # Use device index for CUDA, -1 for CPU
                device=0 if DEVICE == 'cuda' else -1
            )
            print("Sentiment model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Failed to load sentiment model: {e}")

def predict_sentiment(text_list):
    """
    Predicts sentiment for a list of texts.
    Returns a list of dictionaries with 'label' and 'score'.
    """
    if sentiment_pipeline is None:
        load_sentiment_model()
    
    # Handle case where model failed to load
    if sentiment_pipeline is None:
         return [{"label": "NEUTRAL", "score": 0.5}] * len(text_list)

    if not isinstance(text_list, list):
        text_list = [text_list]

    try:
        # batch_size can be tuned for performance based on hardware
        results = sentiment_pipeline(text_list, batch_size=32, truncation=True)
        return results
    except Exception as e:
        print(f"Error during sentiment prediction: {e}")
        return [{"label": "NEUTRAL", "score": 0.5}] * len(text_list)

