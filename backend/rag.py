from models import sentiment_pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from collections import Counter, defaultdict
import nltk
import re
import logging
from typing import Dict, List, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NLTK Data Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

# --- Enhanced Model Initialization ---
SUMMARY_MODEL_NAME = "t5-small"
logger.info("Initializing enhanced AI models...")

try:
    summary_tokenizer = T5Tokenizer.from_pretrained(SUMMARY_MODEL_NAME)
    summary_model = T5ForConditionalGeneration.from_pretrained(SUMMARY_MODEL_NAME)
    
    # Initialize additional models for better analysis
    summarization_pipeline = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn"
    )
    logger.info("Enhanced AI models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load AI models: {e}")
    summary_tokenizer = None
    summary_model = None
    summarization_pipeline = None

def preprocess_reviews(reviews: List[str]) -> List[str]:
    """Enhanced preprocessing with better text cleaning"""
    processed = []
    seen_content = set()
    
    for review in reviews:
        if not review or len(review.strip()) < 10:
            continue
            
        # Clean text
        cleaned = re.sub(r'[^\w\s.,!?-]', '', review)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Skip duplicates
        content_hash = cleaned.lower().replace(' ', '')
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        # Skip if too short or too long
        word_count = len(cleaned.split())
        if 5 <= word_count <= 100:
            processed.append(cleaned)
    
    return processed

def enhanced_topic_extraction(reviews: List[str], sentiments: List[str]) -> Dict:
    """Advanced topic extraction with sentiment awareness"""
    if not reviews:
        return {}
    
    # Get stopwords
    try:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        # Add domain-specific stopwords
        stop_words.update(['product', 'item', 'thing', 'stuff', 'good', 'bad', 'nice', 'great'])
    except:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    topic_sentiments = defaultdict(list)
    topic_contexts = defaultdict(list)
    
    for review, sentiment in zip(reviews, sentiments):
        try:
            # Tokenize and tag
            tokens = nltk.word_tokenize(review.lower())
            tagged = nltk.pos_tag(tokens)
            
            # Extract meaningful terms (nouns, adjectives, verbs)
            meaningful_terms = []
            for word, tag in tagged:
                if (len(word) > 3 and 
                    word not in stop_words and
                    word.isalpha() and
                    tag.startswith(('NN', 'JJ', 'VB'))):
                    meaningful_terms.append(word)
            
            # Extract bigrams for better context
            for i in range(len(meaningful_terms) - 1):
                bigram = f"{meaningful_terms[i]} {meaningful_terms[i+1]}"
                topic_sentiments[bigram].append(sentiment)
                topic_contexts[bigram].append(review[:100])
            
            # Also include single terms
            for term in meaningful_terms:
                topic_sentiments[term].append(sentiment)
                topic_contexts[term].append(review[:100])
                
        except Exception as e:
            logger.warning(f"Error processing review for topics: {e}")
            continue
    
    # Calculate topic scores
    topic_analysis = {}
    for topic, sentiments_list in topic_sentiments.items():
        if len(sentiments_list) >= 2:  # Only topics mentioned multiple times
            positive_count = sentiments_list.count('POSITIVE')
            total_count = len(sentiments_list)
            sentiment_ratio = positive_count / total_count
            
            topic_analysis[topic] = {
                'count': total_count,
                'sentiment': sentiment_ratio,
                'contexts': topic_contexts[topic][:3]  # Sample contexts
            }
    
    # Return top 20 topics by frequency
    return dict(sorted(
        topic_analysis.items(), 
        key=lambda x: x[1]['count'], 
        reverse=True
    )[:20])

def generate_enhanced_summary(reviews: List[str], max_length: int = 200) -> str:
    """Generate summary using multiple approaches for better quality"""
    if not reviews:
        return "No reviews available for analysis."
    
    # Combine reviews into chunks
    combined_text = '. '.join(reviews[:20])  # Use top 20 reviews
    
    try:
        # Try BART summarization first (better quality)
        if summarization_pipeline:
            # Truncate if too long
            if len(combined_text) > 1000:
                combined_text = combined_text[:1000] + "..."
            
            summary_result = summarization_pipeline(
                combined_text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            return summary_result[0]['summary_text']
            
    except Exception as e:
        logger.warning(f"BART summarization failed: {e}")
    
    # Fallback to T5
    try:
        if summary_model and summary_tokenizer:
            prompt = f"summarize: Based on customer reviews, this product "
            inputs = summary_tokenizer.encode(
                prompt + combined_text[:800], 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            )
            
            outputs = summary_model.generate(
                inputs,
                max_length=max_length,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                do_sample=True
            )
            
            summary = summary_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()
            
    except Exception as e:
        logger.warning(f"T5 summarization failed: {e}")
    
    # Ultimate fallback: extractive summary
    return create_extractive_summary(reviews, max_length)

def create_extractive_summary(reviews: List[str], max_length: int) -> str:
    """Create extractive summary from key sentences"""
    if not reviews:
        return "No reviews available."
    
    # Score sentences based on word frequency
    word_freq = Counter()
    all_words = []
    
    for review in reviews[:15]:
        words = [w.lower() for w in nltk.word_tokenize(review) if w.isalpha()]
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    
    # Score sentences
    sentence_scores = []
    for review in reviews[:15]:
        sentences = nltk.sent_tokenize(review)
        for sentence in sentences:
            if len(sentence.split()) > 8:  # Skip very short sentences
                words = [w.lower() for w in nltk.word_tokenize(sentence) if w.isalpha()]
                score = sum(word_freq[word] for word in words) / len(words) if words else 0
                sentence_scores.append((sentence, score))
    
    # Get top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:3]
    summary = ' '.join([sent[0] for sent in top_sentences])
    
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary or "Customer feedback analysis completed."

def extract_pros_and_cons(reviews: List[str], sentiments: List[str]) -> Tuple[List[str], List[str]]:
    """Enhanced pros and cons extraction"""
    pros = []
    cons = []
    
    # Keywords that indicate positive/negative aspects
    positive_indicators = ['love', 'great', 'excellent', 'amazing', 'perfect', 'fantastic', 'awesome', 'wonderful']
    negative_indicators = ['hate', 'terrible', 'awful', 'horrible', 'worst', 'disappointing', 'useless', 'broken']
    
    # Extract from positive reviews
    positive_reviews = [reviews[i] for i, sent in enumerate(sentiments) if sent == 'POSITIVE']
    for review in positive_reviews[:10]:
        sentences = nltk.sent_tokenize(review)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in positive_indicators):
                if len(sentence.split()) > 5 and len(sentence) < 150:
                    # Clean and format
                    clean_sentence = re.sub(r'^[^\w]*', '', sentence).strip()
                    if clean_sentence and clean_sentence not in pros:
                        pros.append(clean_sentence)
                        if len(pros) >= 5:
                            break
        if len(pros) >= 5:
            break
    
    # Extract from negative reviews
    negative_reviews = [reviews[i] for i, sent in enumerate(sentiments) if sent == 'NEGATIVE']
    for review in negative_reviews[:10]:
        sentences = nltk.sent_tokenize(review)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in negative_indicators):
                if len(sentence.split()) > 5 and len(sentence) < 150:
                    clean_sentence = re.sub(r'^[^\w]*', '', sentence).strip()
                    if clean_sentence and clean_sentence not in cons:
                        cons.append(clean_sentence)
                        if len(cons) >= 5:
                            break
        if len(cons) >= 5:
            break
    
    # Fallback: generate generic pros/cons if extraction fails
    if not pros:
        pros = generate_generic_aspects(positive_reviews, "positive")
    if not cons:
        cons = generate_generic_aspects(negative_reviews, "negative")
    
    return pros[:3], cons[:3]  # Limit to top 3 each

def generate_generic_aspects(reviews: List[str], aspect_type: str) -> List[str]:
    """Generate generic aspects when extraction fails"""
    if aspect_type == "positive":
        return [
            "Users found positive aspects in their experience",
            "Some customers were satisfied with the product",
            "Positive feedback was noted in several reviews"
        ]
    else:
        return [
            "Some users reported issues with the product",
            "Certain aspects received negative feedback",
            "Areas for improvement were mentioned by customers"
        ]

def calculate_confidence_score(sentiment_data: Dict, review_count: int) -> str:
    """Calculate confidence based on review volume and sentiment distribution"""
    total_reviews = sentiment_data.get('positive', 0) + sentiment_data.get('negative', 0)
    
    if total_reviews >= 50:
        return "High"
    elif total_reviews >= 20:
        return "Medium" 
    else:
        return "Low"

def analyze_reviews_definitively(reviews: List[str]) -> Dict:
    """Enhanced review analysis with better AI processing"""
    logger.info(f"Analyzing {len(reviews)} reviews")
    
    if not reviews:
        return {
            "summary": "No review content available for analysis.",
            "sentiment": {"positive": 0, "negative": 0},
            "verdict": "Unknown",
            "advantages": ["No data available"],
            "disadvantages": ["No data available"],
            "key_topics": {},
            "confidence": "Low"
        }

    # Preprocess reviews
    processed_reviews = preprocess_reviews(reviews)
    logger.info(f"Processed {len(processed_reviews)} valid reviews")
    
    if not processed_reviews:
        return {
            "summary": "No valid review content found after processing.",
            "sentiment": {"positive": 0, "negative": 0},
            "verdict": "Insufficient Data",
            "advantages": ["No clear advantages identified"],
            "disadvantages": ["No clear disadvantages identified"],
            "key_topics": {},
            "confidence": "Low"
        }

    # Sentiment analysis
    try:
        sentiment_results = sentiment_pipeline(processed_reviews)
        sentiments = [result['label'] for result in sentiment_results]
        sentiment_counts = Counter(sentiments)
        
        positive_count = sentiment_counts.get('POSITIVE', 0)
        negative_count = sentiment_counts.get('NEGATIVE', 0)
        total_count = len(sentiments)
        
        logger.info(f"Sentiment: {positive_count} positive, {negative_count} negative")
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        # Fallback sentiment analysis
        positive_count = len([r for r in processed_reviews if any(word in r.lower() for word in ['good', 'great', 'love', 'excellent'])])
        negative_count = len(processed_reviews) - positive_count
        sentiments = ['POSITIVE'] * positive_count + ['NEGATIVE'] * negative_count
        total_count = len(processed_reviews)

    # Calculate verdict
    positive_percentage = (positive_count / total_count * 100) if total_count > 0 else 0
    
    if positive_percentage >= 70:
        verdict = "Good Buy"
    elif positive_percentage >= 50:
        verdict = "Consider Alternatives"
    elif positive_percentage >= 30:
        verdict = "Mixed Opinions"
    else:
        verdict = "Not Recommended"

    # Generate enhanced summary
    summary = generate_enhanced_summary(processed_reviews)
    
    # Extract pros and cons
    advantages, disadvantages = extract_pros_and_cons(processed_reviews, sentiments)
    
    # Enhanced topic extraction
    key_topics = enhanced_topic_extraction(processed_reviews, sentiments)
    
    # Calculate confidence
    sentiment_data = {"positive": positive_count, "negative": negative_count}
    confidence = calculate_confidence_score(sentiment_data, total_count)
    
    logger.info(f"Analysis complete. Verdict: {verdict}, Confidence: {confidence}")
    
    return {
        "summary": summary,
        "sentiment": sentiment_data,
        "verdict": verdict,
        "advantages": advantages or ["Positive aspects noted by some users"],
        "disadvantages": disadvantages or ["Some areas for improvement identified"],
        "key_topics": key_topics,
        "confidence": confidence,
        "total_reviews_analyzed": total_count
    }

def summarize_and_analyze_comparison(analysis1: Dict, analysis2: Dict) -> str:
    """Enhanced comparison with better reasoning"""
    
    product1_name = analysis1.get('product_name', 'Product 1')
    product2_name = analysis2.get('product_name', 'Product 2')
    
    # Get verdict scores for comparison
    verdict_scores = {
        "Good Buy": 4,
        "Consider Alternatives": 3,
        "Mixed Opinions": 2,
        "Not Recommended": 1,
        "Unknown": 0,
        "Insufficient Data": 0
    }
    
    score1 = verdict_scores.get(analysis1.get('verdict', 'Unknown'), 0)
    score2 = verdict_scores.get(analysis2.get('verdict', 'Unknown'), 0)
    
    # Analyze sentiment ratios
    sent1 = analysis1.get('sentiment', {})
    sent2 = analysis2.get('sentiment', {})
    
    total1 = sent1.get('positive', 0) + sent1.get('negative', 0)
    total2 = sent2.get('positive', 0) + sent2.get('negative', 0)
    
    ratio1 = sent1.get('positive', 0) / total1 if total1 > 0 else 0
    ratio2 = sent2.get('positive', 0) / total2 if total2 > 0 else 0
    
    # Generate comparison
    if score1 > score2 and ratio1 > ratio2:
        winner = product1_name
        reasoning = f"has significantly better user satisfaction ({ratio1:.1%} vs {ratio2:.1%} positive)"
    elif score2 > score1 and ratio2 > ratio1:
        winner = product2_name
        reasoning = f"shows superior customer approval ({ratio2:.1%} vs {ratio1:.1%} positive)"
    elif abs(ratio1 - ratio2) < 0.1:
        winner = "Both products are competitive"
        reasoning = f"with similar satisfaction levels ({ratio1:.1%} vs {ratio2:.1%})"
    else:
        if ratio1 > ratio2:
            winner = product1_name
            reasoning = f"edges ahead with {ratio1:.1%} positive vs {ratio2:.1%}"
        else:
            winner = product2_name
            reasoning = f"leads with {ratio2:.1%} positive vs {ratio1:.1%}"
    
    # Add specific advantages comparison
    adv1_count = len(analysis1.get('advantages', []))
    adv2_count = len(analysis2.get('advantages', []))
    
    comparison_text = f"**Final Recommendation:** {winner} {reasoning}. "
    
    if winner not in ["Both products are competitive"]:
        comparison_text += f"The recommended product offers more consistent positive feedback and fewer reported issues. "
    
    comparison_text += f"Based on analysis of {total1} reviews for {product1_name} and {total2} reviews for {product2_name}."
    
    return comparison_text