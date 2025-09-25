import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available, downloading if necessary.
try:
    stopwords.words('english')
    word_tokenize("test")
except LookupError:
    print("Downloading NLTK resources (stopwords, punkt)...")
    nltk.download('stopwords')
    nltk.download('punkt')
    print("NLTK resources downloaded.")

# Load stopwords once to be efficient
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans and preprocesses a single string of text.
    - Converts to lowercase
    - Removes URLs, special characters, and numbers
    - Removes stopwords
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references, #, and emojis (by removing non-ASCII)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove special characters and numbers, keeping only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin into a single string
    return " ".join(filtered_tokens)

