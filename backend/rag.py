from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from embeddings import find_similar_reviews
import re

# --- Model Configuration ---
SUMMARIZATION_MODEL_NAME = 't5-small'

# --- Global Model Variables ---
summarizer = None
summarizer_tokenizer = None

def load_generative_model():
    """Loads the T5 summarization model and tokenizer into memory."""
    global summarizer, summarizer_tokenizer
    if summarizer is None:
        print(f"Loading generative model ({SUMMARIZATION_MODEL_NAME})...")
        try:
            summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
            summarizer = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
            print("Generative model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Failed to load generative model: {e}")

def generate_text(prompt, max_length=150, min_length=40):
    """Generic function to generate text from a prompt using the T5 model."""
    if summarizer is None:
        load_generative_model()
    
    if summarizer is None:
        return "Generative model is not available."
    
    inputs = summarizer_tokenizer.encode(
        prompt, 
        return_tensors='pt', 
        max_length=1024,
        truncation=True
    )
    
    output_ids = summarizer.generate(
        inputs, 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    
    output_text = summarizer_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def generate_summary(text_chunk):
    """Generates a summary for a given block of text."""
    prompt = f"summarize: {text_chunk}"
    return generate_text(prompt)

def generate_pros_and_cons(reviews_text):
    """
    NEW: Generates a bulleted list of advantages and disadvantages from reviews.
    """
    prompt = (
        "Based on the following customer reviews, identify the main advantages and disadvantages of the product. "
        "Present the output in two distinct lists with bullet points. "
        "Do not add any introductory or concluding sentences. \n\n"
        f"Reviews: \"{reviews_text}\"\n\n"
        "Advantages:\n* \n\nDisadvantages:\n* "
    )
    
    # Use HTML tags for formatting the output
    response = generate_text(prompt, max_length=250, min_length=50)
    
    # Post-process to ensure clean HTML list formatting
    response = response.replace("Advantages:", "<strong>Advantages:</strong><ul>")
    response = response.replace("Disadvantages:", "</ul><strong>Disadvantages:</strong><ul>")
    response = response.replace("* ", "<li>")
    response = response.replace("\n", "</li>") # Replace newlines with closing tags
    response += "</li></ul>" # Add final closing tags
    
    # Clean up any potential double tags
    response = re.sub(r'<li>\s*</li>', '', response) # Remove empty list items
    return response

def answer_query_with_rag(query):
    """
    Answers a user's query using the RAG pipeline.
    1. Finds similar reviews (Retrieval).
    2. Feeds them to a generative model as context (Generation).
    """
    print(f"RAG pipeline initiated for query: '{query}'")
    similar_reviews = find_similar_reviews(query, k=5)
    
    if similar_reviews.empty:
        return "I couldn't find enough information in the local dataset reviews to answer that question."
    
    context = " ".join(similar_reviews['cleaned_text'].tolist())
    
    prompt = (
        f"Please answer the following question based ONLY on the provided customer reviews. "
        f"Do not use any outside knowledge. Summarize the key points from the reviews that are relevant to the question.\n\n"
        f"Question: \"{query}\"\n\n"
        f"Relevant Reviews: {context}\n\n"
        f"Answer:"
    )

    return generate_text(prompt)

