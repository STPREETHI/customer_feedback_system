from transformers import pipeline

# --- Model Configuration ---
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# --- Global Model Instance ---
# We initialize the model once here so it's loaded into memory
# and ready for use by the entire application.
try:
    print("--- Loading Sentiment Analysis Model... This may take a moment. ---")
    # Use framework="pt" to explicitly specify PyTorch and avoid TensorFlow conflicts
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_NAME,
        framework="pt" 
    )
    print("--- Sentiment Model loaded successfully. ---")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load sentiment model: {e}")
    sentiment_pipeline = None

