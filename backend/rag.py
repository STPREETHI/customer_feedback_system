from models import sentiment_pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from collections import Counter
import nltk
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- NLTK Data Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    log.info("--- First-time setup: Downloading NLTK data... ---")
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

# --- Model Initialization ---
SUMMARY_MODEL_NAME = "t5-small"
log.info("--- Initializing RAG models (T5-small)... ---")
summary_tokenizer = T5Tokenizer.from_pretrained(SUMMARY_MODEL_NAME)
summary_model = T5ForConditionalGeneration.from_pretrained(SUMMARY_MODEL_NAME)
log.info("--- RAG models initialized. ---")

def generate_with_t5(prompt: str, max_length: int = 150) -> str:
    """A robust function to generate text from a prompt and polish the output."""
    inputs = summary_tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    output_ids = summary_model.generate(
        inputs, max_length=max_length, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    decoded_text = summary_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if decoded_text:
        return decoded_text[0].upper() + decoded_text[1:]
    return decoded_text

def analyze_reviews_definitively(reviews: list):
    """
    The definitive, multi-step analysis engine. It breaks down the AI task into
    separate, focused prompts for higher quality and reliability.
    """
    if not reviews:
        return {"summary": "No review content found.", "sentiment": {}, "verdict": "Unknown", "advantages": [], "disadvantages": [], "key_topics": {}}

    # --- Foundational NLP Analysis ---
    sentiments_result = sentiment_pipeline(reviews)
    overall_sentiments = [s['label'] for s in sentiments_result]
    overall_counts = Counter(overall_sentiments)
    positive_percentage = float((overall_counts.get('POSITIVE', 0) / len(overall_sentiments)) * 100 if overall_sentiments else 0)
    
    # --- Sentiment-Aware Topic Extraction ---
    stop_words = set(nltk.corpus.stopwords.words('english'))
    topic_sentiments = {}
    for i, review in enumerate(reviews):
        words = nltk.word_tokenize(review)
        tagged_words = nltk.pos_tag(words)
        sentiment = sentiments_result[i]['label']
        for word, tag in tagged_words:
            if (tag.startswith('NN') or tag.startswith('JJ')) and word.lower() not in stop_words and len(word) > 3:
                topic = word.lower()
                if topic not in topic_sentiments: topic_sentiments[topic] = []
                topic_sentiments[topic].append(sentiment)
    
    key_topics_for_cloud = {}
    for topic, sentiments in topic_sentiments.items():
        if len(sentiments) > 1:
            positive_ratio = Counter(sentiments).get('POSITIVE', 0) / len(sentiments)
            key_topics_for_cloud[topic] = {"count": len(sentiments), "sentiment": positive_ratio}
    top_topics = {k: v for k, v in sorted(key_topics_for_cloud.items(), key=lambda item: item[1]['count'], reverse=True)[:15]}

    # --- Generate Verdict ---
    verdict = "Mixed Opinions"
    if positive_percentage >= 65: verdict = "Good Buy"
    elif positive_percentage >= 40: verdict = "Consider Alternatives"
    else: verdict = "Not Recommended"

    # --- Multi-Step AI Content Generation ---
    combined_reviews_text = " ".join(reviews[:30])
    
    log.info("Generating summary...")
    summary_prompt = f"Act as an expert product analyst. Based on these reviews, write a concise summary. Use professional language. Reviews: \"{combined_reviews_text}\""
    summary = generate_with_t5(summary_prompt, max_length=150)

    log.info("Generating advantages...")
    adv_prompt = f"Based on these reviews, list three key advantages of the product in a bulleted list. Use proper capitalization. Reviews: \"{combined_reviews_text}\""
    advantages_text = generate_with_t5(adv_prompt, max_length=100)
    advantages = [adv[1] for adv in re.findall(r'(\d+\.\s*|-\s*|\*\s*)([^\n]+)', advantages_text)]

    log.info("Generating disadvantages...")
    dis_prompt = f"Based on these reviews, list three key disadvantages of the product in a bulleted list. Use proper capitalization. Reviews: \"{combined_reviews_text}\""
    disadvantages_text = generate_with_t5(dis_prompt, max_length=100)
    disadvantages = [dis[1] for dis in re.findall(r'(\d+\.\s*|-\s*|\*\s*)([^\n]+)', disadvantages_text)]
    
    log.info(f"Enhanced analysis complete. Verdict: {verdict}")

    return {
        "summary": summary,
        "sentiment": {"positive": overall_counts.get('POSITIVE', 0), "negative": overall_counts.get('NEGATIVE', 0)},
        "verdict": verdict,
        "advantages": advantages if advantages else ["No specific advantages were consistently identified."],
        "disadvantages": disadvantages if disadvantages else ["No specific disadvantages were consistently identified."],
        "key_topics": top_topics
    }

def summarize_and_analyze_comparison(analysis1: dict, analysis2: dict):
    """Generates a detailed comparison summary."""
    log.info("Generating comparison summary...")
    prompt = f"""
    You are a product comparison expert. Compare two products.
    
    Product 1: "{analysis1['product_name']}" is rated {analysis1['verdict']}.
    - Pros: {', '.join(analysis1['advantages'])}
    - Cons: {', '.join(analysis1['disadvantages'])}

    Product 2: "{analysis2['product_name']}" is rated {analysis2['verdict']}.
    - Pros: {', '.join(analysis2['advantages'])}
    - Cons: {', '.join(analysis2['disadvantages'])}

    Verdict: Which product is the better buy and why? Be definitive and concise.
    """
    return generate_with_t5(prompt, max_length=250)

