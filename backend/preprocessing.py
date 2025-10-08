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
    Enhanced text cleaning with proper special character removal and keyword preservation.
    """
    if not isinstance(text, str):
        return ""
    
    if len(text.strip()) == 0:
        return ""

    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters but keep basic punctuation and spaces
    # Keep: letters, numbers, spaces, periods, commas, exclamation, question marks
    text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
    
    # Remove extra dots and punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # Remove numbers but preserve important ones (ratings, versions)
    # Keep patterns like "4.5", "v2.0", "iPhone 15"
    text = re.sub(r'\b\d+\b(?!\s*(star|rating|version|pro|max|plus|mini))', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very short and very long words
    words = text.split()
    filtered_words = []
    
    for word in words:
        # Keep words between 2-15 characters
        if 2 <= len(word) <= 15:
            # Skip if it's just punctuation
            if not re.match(r'^[\.\,\!\?]+$', word):
                filtered_words.append(word)
    
    # Rejoin words
    text = ' '.join(filtered_words)
    
    # Remove stopwords but keep sentiment-important ones
    important_words = {'not', 'no', 'never', 'nothing', 'very', 'really', 'quite', 'too', 'much', 'more', 'most', 'best', 'worst', 'good', 'bad', 'great', 'terrible'}
    
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words or word in important_words]
    
    # Final cleanup
    result = ' '.join(filtered_tokens).strip()
    
    # Return only if meaningful content remains
    return result if len(result.split()) >= 3 else ""