from models import sentiment_pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from collections import Counter
import nltk
import re
import logging
import random

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

def keyword_based_sentiment_analysis(reviews):
    """
    Advanced keyword-based sentiment analysis with scoring and context awareness.
    Returns percentages instead of raw counts.
    """
    # Extended keyword dictionaries with weights
    positive_keywords = {
        # Highly positive (weight 3)
        'excellent': 3, 'amazing': 3, 'outstanding': 3, 'perfect': 3, 'fantastic': 3,
        'brilliant': 3, 'exceptional': 3, 'superb': 3, 'magnificent': 3,
        
        # Very positive (weight 2.5)
        'wonderful': 2.5, 'awesome': 2.5, 'incredible': 2.5, 'remarkable': 2.5,
        'impressive': 2.5, 'beautiful': 2.5, 'stunning': 2.5,
        
        # Positive (weight 2)
        'great': 2, 'good': 2, 'nice': 2, 'love': 2, 'lovely': 2, 'pleased': 2,
        'happy': 2, 'satisfied': 2, 'recommend': 2, 'recommended': 2, 'solid': 2,
        
        # Moderately positive (weight 1.5)
        'decent': 1.5, 'fine': 1.5, 'okay': 1.5, 'useful': 1.5, 'helpful': 1.5,
        'convenient': 1.5, 'comfortable': 1.5, 'reliable': 1.5, 'sturdy': 1.5,
        
        # Mildly positive (weight 1)
        'quality': 1, 'fast': 1, 'quick': 1, 'easy': 1, 'smooth': 1, 'clear': 1,
        'bright': 1, 'clean': 1, 'fresh': 1, 'works': 1, 'working': 1
    }
    
    negative_keywords = {
        # Highly negative (weight -3)
        'terrible': -3, 'awful': -3, 'horrible': -3, 'disgusting': -3, 'atrocious': -3,
        'appalling': -3, 'dreadful': -3, 'abysmal': -3, 'deplorable': -3,
        
        # Very negative (weight -2.5)
        'pathetic': -2.5, 'ridiculous': -2.5, 'outrageous': -2.5, 'unacceptable': -2.5,
        'disaster': -2.5, 'nightmare': -2.5, 'catastrophe': -2.5,
        
        # Negative (weight -2)
        'bad': -2, 'poor': -2, 'hate': -2, 'disappointed': -2, 'disappointing': -2,
        'useless': -2, 'worthless': -2, 'broken': -2, 'defective': -2, 'faulty': -2,
        'waste': -2, 'regret': -2, 'worst': -2,
        
        # Moderately negative (weight -1.5)
        'annoying': -1.5, 'frustrating': -1.5, 'confusing': -1.5, 'complicated': -1.5,
        'uncomfortable': -1.5, 'inconvenient': -1.5, 'unreliable': -1.5,
        
        # Mildly negative (weight -1)
        'slow': -1, 'expensive': -1, 'costly': -1, 'difficult': -1, 'hard': -1,
        'problem': -1, 'issue': -1, 'trouble': -1, 'concern': -1, 'lacking': -1,
        'missing': -1, 'limited': -1
    }
    
    # Negation words that flip sentiment
    negation_words = {'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none', 
                     'neither', 'nor', 'hardly', 'barely', 'scarcely', 'seldom', 'rarely'}
    
    # Intensifier words that amplify sentiment
    intensifiers = {
        'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 2.0,
        'completely': 1.8, 'totally': 1.8, 'quite': 1.3, 'rather': 1.2, 'pretty': 1.2,
        'so': 1.4, 'too': 1.3, 'highly': 1.6, 'deeply': 1.5, 'truly': 1.4
    }
    
    total_positive = 0
    total_negative = 0
    sentiment_words_found = {}
    total_reviews_analyzed = len([r for r in reviews if r and len(r.strip()) >= 3])
    
    for review_text in reviews:
        if not review_text or len(review_text.strip()) < 3:
            continue
            
        words = review_text.lower().split()
        
        i = 0
        while i < len(words):
            word = words[i].strip('.,!?;:"')
            
            # Check for negation in the previous 2 words
            negated = False
            intensifier_mult = 1.0
            
            # Look back for negations and intensifiers
            for j in range(max(0, i-2), i):
                prev_word = words[j].strip('.,!?;:"')
                if prev_word in negation_words:
                    negated = True
                if prev_word in intensifiers:
                    intensifier_mult = intensifiers[prev_word]
            
            # Process positive keywords
            if word in positive_keywords:
                score = positive_keywords[word] * intensifier_mult
                if negated:
                    score = -score  # Flip sentiment for negation
                    total_negative += abs(score)
                    sentiment_words_found[f"not_{word}"] = sentiment_words_found.get(f"not_{word}", 0) + 1
                else:
                    total_positive += score
                    sentiment_words_found[word] = sentiment_words_found.get(word, 0) + 1
            
            # Process negative keywords
            elif word in negative_keywords:
                score = abs(negative_keywords[word]) * intensifier_mult
                if negated:
                    score = score  # Double negation makes it positive
                    total_positive += score
                    sentiment_words_found[f"not_{word}"] = sentiment_words_found.get(f"not_{word}", 0) + 1
                else:
                    total_negative += score
                    sentiment_words_found[word] = sentiment_words_found.get(word, 0) + 1
            
            i += 1
    
    # Calculate percentages instead of raw counts
    total_sentiment_score = total_positive + total_negative
    
    if total_sentiment_score > 0:
        positive_percentage = (total_positive / total_sentiment_score) * 100
        negative_percentage = (total_negative / total_sentiment_score) * 100
    else:
        positive_percentage = 50
        negative_percentage = 50
    
    # Convert to readable format
    positive_display = f"{positive_percentage:.1f}%"
    negative_display = f"{negative_percentage:.1f}%"
    
    return {
        "positive": int(positive_percentage),  # For chart display
        "negative": int(negative_percentage),  # For chart display  
        "positive_display": positive_display,  # For text display
        "negative_display": negative_display,  # For text display
        "total_reviews": total_reviews_analyzed,
        "sentiment_words": sentiment_words_found,
        "total_score": total_positive - total_negative
    }

def extract_keyword_based_topics(reviews):
    """
    Extract key topics using keyword frequency and context analysis.
    """
    # Product-specific topic keywords
    topic_keywords = {
        'quality': ['quality', 'build', 'construction', 'material', 'solid', 'sturdy', 'durable', 'cheap', 'flimsy'],
        'performance': ['fast', 'slow', 'speed', 'performance', 'lag', 'smooth', 'responsive', 'quick'],
        'design': ['design', 'look', 'appearance', 'beautiful', 'ugly', 'style', 'color', 'size'],
        'usability': ['easy', 'difficult', 'user', 'interface', 'simple', 'complicated', 'intuitive'],
        'value': ['price', 'cost', 'expensive', 'cheap', 'worth', 'value', 'money', 'affordable'],
        'reliability': ['reliable', 'unreliable', 'stable', 'crash', 'bug', 'issue', 'problem', 'works'],
        'customer_service': ['service', 'support', 'help', 'staff', 'customer', 'response', 'assistance']
    }
    
    topic_scores = {}
    topic_sentiments = {}
    
    # Analyze sentiment for each topic
    sentiment_result = keyword_based_sentiment_analysis(reviews)
    sentiment_words = sentiment_result.get('sentiment_words', {})
    
    for topic, keywords in topic_keywords.items():
        topic_count = 0
        topic_sentiment_score = 0
        
        for review in reviews:
            review_lower = review.lower()
            for keyword in keywords:
                if keyword in review_lower:
                    topic_count += review_lower.count(keyword)
                    
                    # Calculate sentiment for this keyword occurrence
                    if keyword in sentiment_words:
                        # Get context around keyword for sentiment
                        words = review_lower.split()
                        for i, word in enumerate(words):
                            if keyword in word:
                                # Check sentiment in surrounding context (±3 words)
                                context = ' '.join(words[max(0,i-3):min(len(words),i+4)])
                                context_sentiment = keyword_based_sentiment_analysis([context])
                                if context_sentiment['positive'] > context_sentiment['negative']:
                                    topic_sentiment_score += 1
                                else:
                                    topic_sentiment_score -= 1
        
        if topic_count > 0:
            sentiment_ratio = (topic_sentiment_score + topic_count) / (2 * topic_count)  # Normalize to 0-1
            sentiment_ratio = max(0, min(1, sentiment_ratio))  # Clamp to 0-1
            
            topic_scores[topic] = {
                "count": min(topic_count, 20),  # Cap for visualization
                "sentiment": sentiment_ratio
            }
    
    # Add most frequent individual words as topics
    all_words = ' '.join(reviews).lower().split()
    word_freq = Counter(all_words)
    
    # Filter for meaningful words
    meaningful_words = {}
    for word, count in word_freq.most_common(50):
        if (len(word) >= 4 and count >= 2 and 
            word not in ['this', 'that', 'with', 'have', 'been', 'were', 'they', 'from', 'would']):
            
            # Calculate sentiment for this word
            word_sentiment = 0.5  # Neutral default
            if word in sentiment_words:
                word_reviews = [r for r in reviews if word in r.lower()]
                if word_reviews:
                    word_sent_analysis = keyword_based_sentiment_analysis(word_reviews)
                    total = word_sent_analysis['positive'] + word_sent_analysis['negative']
                    if total > 0:
                        word_sentiment = word_sent_analysis['positive'] / total
            
            meaningful_words[word] = {
                "count": min(count, 15),
                "sentiment": word_sentiment
            }
    
    # Combine topic keywords and individual words
    topic_scores.update(list(meaningful_words.items())[:10])  # Add top 10 individual words
    
    return topic_scores

def generate_with_t5(prompt: str, max_length: int = 100) -> str:
    """Fast T5 generation with optimized settings."""
    inputs = summary_tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True)
    output_ids = summary_model.generate(
        inputs, max_length=max_length, min_length=15, length_penalty=1.5, 
        num_beams=2, early_stopping=True, do_sample=False
    )
    decoded_text = summary_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded_text.strip().capitalize() if decoded_text else ""

def smart_snippet_selection(reviews: list, max_snippets: int = 20) -> list:
    """Ultra-fast snippet selection for quick processing."""
    if len(reviews) <= max_snippets:
        return reviews
    
    # Quick selection: take every nth review to get diverse sample
    step = len(reviews) // max_snippets
    selected = reviews[::step][:max_snippets]
    
    # Add few longest reviews for quality
    sorted_by_length = sorted(reviews, key=len, reverse=True)[:5]
    selected.extend(sorted_by_length)
    
    return list(set(selected))[:max_snippets]

def clean_and_fix_grammar(text: str) -> str:
    """Clean text and fix basic grammar issues."""
    if not text:
        return ""
    
    # Remove repetitive patterns
    text = re.sub(r'(\b\w+\b)(\s+\1\b)+', r'\1', text)  # Remove word repetitions
    text = re.sub(r'(\b\w+\s+\w+)\s+\1', r'\1', text)  # Remove phrase repetitions
    
    # Fix common grammar patterns
    text = re.sub(r'\bthis\s+this\b', 'this', text)
    text = re.sub(r'\bthe\s+the\b', 'the', text)
    text = re.sub(r'\band\s+and\b', 'and', text)
    
    # Remove special characters and clean up
    text = re.sub(r'[^\w\s\.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Ensure proper capitalization
    if text:
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
    # Ensure proper ending
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
        
    return text

def generate_keyword_based_summary(reviews: list, sentiment_data: dict) -> str:
    """Generate summary based on keyword analysis with percentage display."""
    if not reviews:
        return "No user feedback available for comprehensive analysis."
    
    total_reviews = sentiment_data.get("total_reviews", len(reviews))
    positive_pct = sentiment_data.get("positive_display", "50.0%")
    negative_pct = sentiment_data.get("negative_display", "50.0%")
    positive_ratio = sentiment_data["positive"] / 100  # Convert back to ratio for logic
    
    # Get most common sentiment words
    sentiment_words = sentiment_data.get('sentiment_words', {})
    top_positive = [word for word, count in Counter(sentiment_words).most_common(3) 
                   if not word.startswith('not_')]
    top_negative = [word for word, count in Counter(sentiment_words).most_common(3) 
                   if word.startswith('not_') or word in ['bad', 'poor', 'terrible', 'awful', 'disappointing']]
    
    # Generate template-based summary with percentages
    if positive_ratio >= 0.7:
        summary = f"Users express positive feedback with {positive_pct} positive sentiment across {total_reviews} reviews analyzed. "
        if top_positive:
            summary += f"Commonly mentioned positives include {', '.join(top_positive[:2])}. "
        summary += "Overall satisfaction appears high based on user sentiment."
        
    elif positive_ratio >= 0.4:
        summary = f"Mixed user opinions found with {positive_pct} positive and {negative_pct} negative sentiment across {total_reviews} reviews. "
        if top_positive and top_negative:
            summary += f"Users appreciate {', '.join(top_positive[:1])} but note concerns about {', '.join(top_negative[:1])}. "
        summary += "Consider individual needs when evaluating this product."
        
    else:
        summary = f"User feedback shows concerns with {negative_pct} negative sentiment across {total_reviews} reviews analyzed. "
        if top_negative:
            summary += f"Common issues mentioned include {', '.join(top_negative[:2])}. "
        summary += "Careful consideration recommended before purchase."
    
    return clean_and_fix_grammar(summary)

def extract_keyword_based_advantages(reviews: list, sentiment_data: dict) -> list:
    """Extract advantages using keyword analysis."""
    if not reviews:
        return ["Positive aspects noted in user feedback", "Good user experience reported", "Recommended by users"]
    
    # Define advantage patterns
    advantage_patterns = {
        'quality': "Good build quality and materials noted by users",
        'fast': "Fast performance and quick response times mentioned", 
        'easy': "User-friendly and easy to use according to reviews",
        'great': "Great overall experience reported by users",
        'love': "Users express strong satisfaction and recommendation",
        'recommend': "Highly recommended by satisfied customers",
        'excellent': "Excellent quality and performance highlighted",
        'amazing': "Amazing features and capabilities praised",
        'perfect': "Perfect fit for user needs and expectations",
        'good': "Good value and reliable performance noted",
        'works': "Reliable functionality confirmed by users",
        'solid': "Solid construction and dependable operation",
        'beautiful': "Attractive design and aesthetic appeal mentioned",
        'comfortable': "Comfortable usage experience reported"
    }
    
    sentiment_words = sentiment_data.get('sentiment_words', {})
    advantages = []
    
    # Find advantages based on positive sentiment words
    for word, count in sentiment_words.items():
        if not word.startswith('not_') and word in advantage_patterns and count > 0:
            advantages.append(advantage_patterns[word])
    
    # Add generic advantages if specific ones not found
    if len(advantages) < 3:
        generic_advantages = [
            "Positive user experiences reported in reviews",
            "Good value for money mentioned by users", 
            "Reliable performance noted in feedback",
            "User-friendly design appreciated",
            "Quality construction highlighted",
            "Effective functionality confirmed"
        ]
        advantages.extend(generic_advantages)
    
    return advantages[:3]

def extract_keyword_based_disadvantages(reviews: list, sentiment_data: dict) -> list:
    """Extract disadvantages using keyword analysis."""
    if not reviews:
        return ["Some concerns raised in user feedback", "Mixed experiences reported", "Individual preferences may vary"]
    
    # Define disadvantage patterns
    disadvantage_patterns = {
        'expensive': "Price point considered high by some users",
        'slow': "Performance speed concerns mentioned in reviews",
        'difficult': "Usability challenges noted by some users", 
        'bad': "Quality issues reported by users",
        'poor': "Poor performance or build quality mentioned",
        'problem': "Technical problems and issues reported",
        'issue': "Various issues and concerns raised by users",
        'not_good': "Not meeting user expectations in some cases",
        'not_great': "Performance below expectations for some users",
        'not_recommend': "Some users would not recommend this product",
        'disappointing': "Disappointing experience reported by users",
        'waste': "Poor value for money according to some reviews",
        'broken': "Durability and reliability concerns mentioned",
        'terrible': "Significant quality or performance issues noted",
        'awful': "Very poor user experience reported"
    }
    
    sentiment_words = sentiment_data.get('sentiment_words', {})
    disadvantages = []
    
    # Find disadvantages based on negative sentiment words
    for word, count in sentiment_words.items():
        if word in disadvantage_patterns and count > 0:
            disadvantages.append(disadvantage_patterns[word])
    
    # Add generic disadvantages if specific ones not found
    if len(disadvantages) < 3:
        generic_disadvantages = [
            "Price concerns mentioned by some users",
            "Minor functionality issues reported",
            "Mixed satisfaction levels in reviews",
            "Some usability challenges noted",
            "Individual preferences may vary",
            "Limited features mentioned by users"
        ]
        disadvantages.extend(generic_disadvantages)
    
    return disadvantages[:3]

def analyze_reviews_definitively(reviews: list):
    """Ultra-optimized analysis pipeline with keyword-based processing."""
    if not reviews:
        return {
            "summary": "No review content found.",
            "sentiment": {"positive": 0, "negative": 0},
            "verdict": "Unknown",
            "advantages": [],
            "disadvantages": [],
            "key_topics": {}
        }

    original_count = len(reviews)
    # Aggressive optimization: limit to 30 snippets max
    reviews = smart_snippet_selection(reviews, max_snippets=30)
    log.info(f"Keyword-optimized: {original_count} → {len(reviews)} snippets")

    # Use keyword-based analysis instead of AI models
    sentiment_data = keyword_based_sentiment_analysis(reviews)
    key_topics = extract_keyword_based_topics(reviews)
    
    # Generate verdict based on sentiment ratio and total score
    total_sentiment = sentiment_data["positive"] + sentiment_data["negative"]
    positive_percentage = (sentiment_data["positive"] / total_sentiment * 100) if total_sentiment > 0 else 50
    total_score = sentiment_data.get("total_score", 0)
    
    # More nuanced verdict calculation
    if positive_percentage >= 75 and total_score > 5: 
        verdict = "Good Buy"
    elif positive_percentage >= 60 and total_score > 0: 
        verdict = "Consider Alternatives"
    elif positive_percentage >= 35: 
        verdict = "Mixed Opinions"
    else: 
        verdict = "Not Recommended"

    # Generate keyword-based content
    summary = generate_keyword_based_summary(reviews, sentiment_data)
    advantages = extract_keyword_based_advantages(reviews, sentiment_data)
    disadvantages = extract_keyword_based_disadvantages(reviews, sentiment_data)
    
    log.info(f"Keyword-based analysis complete. Verdict: {verdict}")

    return {
        "summary": summary,
        "sentiment": {
            "positive": sentiment_data["positive"], 
            "negative": sentiment_data["negative"],
            "positive_display": sentiment_data.get("positive_display", f"{sentiment_data['positive']}%"),
            "negative_display": sentiment_data.get("negative_display", f"{sentiment_data['negative']}%"),
            "total_reviews": sentiment_data.get("total_reviews", len(reviews))
        },
        "verdict": verdict,
        "advantages": advantages,
        "disadvantages": disadvantages,
        "key_topics": key_topics
    }

def generate_structured_comparison(analysis1: dict, analysis2: dict) -> str:
    """Generate professional comparison summary."""
    product1_name = analysis1.get('product_name', 'Product 1')
    product2_name = analysis2.get('product_name', 'Product 2')
    
    score1 = calculate_product_score(analysis1)
    score2 = calculate_product_score(analysis2)
    score_diff = abs(score1 - score2)
    
    if score_diff > 1.0:
        if score1 > score2:
            return f"{product1_name} significantly outperforms {product2_name} based on user satisfaction and feedback quality. Users report fewer issues and higher overall satisfaction."
        else:
            return f"{product2_name} significantly outperforms {product1_name} based on user satisfaction and feedback quality. Users report fewer issues and higher overall satisfaction."
    elif score_diff > 0.5:
        winner = product1_name if score1 > score2 else product2_name
        return f"{winner} has a moderate advantage in user satisfaction and overall rating compared to its competitor."
    else:
        return f"Both {product1_name} and {product2_name} show comparable performance with similar user satisfaction levels and mixed feedback."

def generate_clear_recommendation(analysis1: dict, analysis2: dict) -> str:
    """Generate clear, decisive recommendation."""
    score1 = calculate_product_score(analysis1)
    score2 = calculate_product_score(analysis2)
    
    product1_name = analysis1.get('product_name', 'Product 1')
    product2_name = analysis2.get('product_name', 'Product 2')
    
    score_diff = abs(score1 - score2)
    
    if score_diff > 1.0:
        winner = analysis1 if score1 > score2 else analysis2
        return f"**Recommended Choice:** {winner['product_name']} - Clear winner with {winner['verdict'].lower()} rating and significantly better user feedback."
    elif score_diff > 0.5:
        winner = analysis1 if score1 > score2 else analysis2
        return f"**Recommended Choice:** {winner['product_name']} - Better overall rating ({winner['verdict'].lower()}) and slightly more positive user sentiment."
    else:
        # Close comparison - provide nuanced recommendation
        if analysis1['verdict'] == analysis2['verdict']:
            return f"**Similar Options:** Both products are comparable. Choose {product1_name} for specific use cases or {product2_name} based on personal preference and availability."
        else:
            better_verdict = analysis1 if calculate_product_score(analysis1) >= calculate_product_score(analysis2) else analysis2
            return f"**Slight Edge:** {better_verdict['product_name']} with {better_verdict['verdict'].lower()} rating, though both options are viable depending on your specific needs."

def summarize_and_analyze_comparison(analysis1: dict, analysis2: dict):
    """Generate structured comparison with professional formatting."""
    log.info("Generating professional comparison analysis...")
    
    comparison_summary = generate_structured_comparison(analysis1, analysis2)
    recommendation = generate_clear_recommendation(analysis1, analysis2)
    
    return {
        "comparison_summary": comparison_summary,
        "recommendation": recommendation
    }

def calculate_product_score(analysis: dict) -> float:
    """Calculate overall product score for comparison."""
    verdict_scores = {
        "Good Buy": 4.0,
        "Consider Alternatives": 2.5,
        "Mixed Opinions": 2.0,
        "Not Recommended": 1.0
    }
    
    base_score = verdict_scores.get(analysis['verdict'], 2.0)
    
    # Adjust based on sentiment ratio
    sentiment = analysis['sentiment']
    total = sentiment['positive'] + sentiment['negative']
    if total > 0:
        sentiment_ratio = sentiment['positive'] / total
        base_score += sentiment_ratio * 1.0
    
    return base_score

def detect_product_category(product_name: str) -> str:
    """Detect product category for suggestions."""
    name_lower = product_name.lower()
    
    # Electronics
    if any(word in name_lower for word in ['iphone', 'galaxy', 'pixel', 'phone', 'smartphone']):
        return 'phone'
    elif any(word in name_lower for word in ['macbook', 'laptop', 'notebook', 'thinkpad', 'computer']):
        return 'laptop'
    elif any(word in name_lower for word in ['ipad', 'tablet', 'surface']):
        return 'tablet'
    elif any(word in name_lower for word in ['watch', 'smartwatch', 'fitbit']):
        return 'smartwatch'
    elif any(word in name_lower for word in ['headphones', 'earbuds', 'airpods', 'speaker', 'audio']):
        return 'audio'
    elif any(word in name_lower for word in ['tv', 'television', 'monitor', 'display']):
        return 'display'
    elif any(word in name_lower for word in ['camera', 'lens', 'dslr', 'gopro']):
        return 'camera'
    
    # Fashion & Clothing
    elif any(word in name_lower for word in ['dress', 'gown', 'frock', 'sundress', 'maxi dress']):
        return 'dress'
    elif any(word in name_lower for word in ['shirt', 'blouse', 'top', 'tshirt', 't-shirt', 'polo']):
        return 'clothing'
    elif any(word in name_lower for word in ['jeans', 'pants', 'trousers', 'shorts', 'leggings']):
        return 'clothing'
    elif any(word in name_lower for word in ['shoes', 'sneakers', 'boots', 'sandals', 'heels', 'footwear']):
        return 'footwear'
    elif any(word in name_lower for word in ['jacket', 'coat', 'hoodie', 'sweater', 'cardigan']):
        return 'clothing'
    elif any(word in name_lower for word in ['bag', 'purse', 'backpack', 'handbag', 'wallet']):
        return 'accessories'
    
    # Home & Kitchen
    elif any(word in name_lower for word in ['mattress', 'pillow', 'bedsheet', 'blanket', 'bed']):
        return 'home'
    elif any(word in name_lower for word in ['microwave', 'oven', 'refrigerator', 'blender', 'toaster']):
        return 'appliances'
    elif any(word in name_lower for word in ['sofa', 'chair', 'table', 'desk', 'furniture']):
        return 'furniture'
    
    # Beauty & Personal Care
    elif any(word in name_lower for word in ['lipstick', 'makeup', 'foundation', 'mascara', 'cosmetics']):
        return 'beauty'
    elif any(word in name_lower for word in ['shampoo', 'conditioner', 'cream', 'lotion', 'skincare']):
        return 'beauty'
    
    # Health & Personal Care
    elif any(word in name_lower for word in ['condom', 'contraceptive', 'lubricant', 'pregnancy test']):
        return 'health'
    elif any(word in name_lower for word in ['vitamin', 'supplement', 'medicine', 'medication', 'pills']):
        return 'health'
    elif any(word in name_lower for word in ['toothbrush', 'toothpaste', 'mouthwash', 'dental']):
        return 'health'
    elif any(word in name_lower for word in ['soap', 'sanitizer', 'deodorant', 'perfume']):
        return 'personal_care'
    
    # Sports & Fitness
    elif any(word in name_lower for word in ['gym', 'fitness', 'yoga', 'exercise', 'workout', 'sports']):
        return 'fitness'
    
    # Books & Media
    elif any(word in name_lower for word in ['book', 'novel', 'textbook', 'kindle', 'ebook']):
        return 'books'
    elif any(word in name_lower for word in ['game', 'gaming', 'xbox', 'playstation', 'nintendo']):
        return 'gaming'
    
    # Automotive
    elif any(word in name_lower for word in ['car', 'vehicle', 'auto', 'tire', 'automotive']):
        return 'automotive'
    
    # Food & Beverages
    elif any(word in name_lower for word in ['coffee', 'tea', 'wine', 'beer', 'food', 'snack']):
        return 'food'
    
    else:
        return 'general'

def extract_competitor_mentions(reviews: list, current_product: str) -> list:
    """Extract competitor/alternative products mentioned in reviews."""
    competitors = []
    current_product_clean = re.sub(r'[^\w\s]', '', current_product.lower())
    
    for review in reviews:
        review_lower = review.lower()
        
        # Look for comparison patterns
        comparison_patterns = [
            r'better than (\w+(?:\s+\w+)*)',
            r'prefer (\w+(?:\s+\w+)*)',
            r'recommend (\w+(?:\s+\w+)*)',
            r'try (\w+(?:\s+\w+)*) instead',
            r'switch to (\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*) is better',
            r'(\w+(?:\s+\w+)*) works better',
            r'go with (\w+(?:\s+\w+)*)'
        ]
        
        for pattern in comparison_patterns:
            matches = re.findall(pattern, review_lower)
            for match in matches:
                if match and len(match) > 2 and match not in current_product_clean:
                    competitors.append(match.strip().title())
    
    return competitors

def find_best_in_category_from_reviews(query: str, reviews: list) -> dict:
    """Find the best product in category based on what users mention in reviews."""
    if not reviews:
        return {
            'name': 'No recommendations found',
            'reason': 'Insufficient review data to identify alternatives.',
            'rating': 'N/A'
        }
    
    # Extract competitor mentions from reviews
    competitors = extract_competitor_mentions(reviews, query)
    
    if competitors:
        # Count mentions and find most recommended
        competitor_counts = Counter(competitors)
        top_competitor = competitor_counts.most_common(1)[0]
        
        # Analyze sentiment around the top competitor
        positive_mentions = 0
        total_mentions = 0
        
        for review in reviews:
            if top_competitor[0].lower() in review.lower():
                total_mentions += 1
                # Simple sentiment check around the mention
                context = review.lower()
                if any(word in context for word in ['better', 'best', 'prefer', 'recommend', 'love', 'amazing', 'excellent']):
                    positive_mentions += 1
        
        if positive_mentions > 0:
            rating = min(5.0, (positive_mentions / max(total_mentions, 1)) * 5.0)
            
            return {
                'name': top_competitor[0],
                'reason': f'Mentioned {top_competitor[1]} times in reviews as a better alternative with positive user feedback.',
                'rating': f'{rating:.1f}/5'
            }
    
    # If no clear competitors, look for general positive brand mentions
    brand_mentions = []
    for review in reviews:
        # Find any brand names mentioned positively
        positive_patterns = [
            r'recommend\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'try\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+better',
            r'prefer\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in positive_patterns:
            matches = re.findall(pattern, review)
            for match in matches:
                if len(match) > 2 and match.lower() not in query.lower():
                    brand_mentions.append(match.strip())
    
    if brand_mentions:
        top_brand = Counter(brand_mentions).most_common(1)[0]
        return {
            'name': top_brand[0],
            'reason': f'Mentioned {top_brand[1]} times as a recommended alternative in user reviews.',
            'rating': '4.0/5'
        }
    
    # Last resort - category-based generic suggestion
    category = detect_product_category(query)
    return {
        'name': f'Research top {category} brands',
        'reason': 'Reviews suggest comparing multiple options in this category for best results.',
        'rating': 'N/A'
    }

def generate_context_based_suggestions(query: str, scraped_data: dict = None) -> dict:
    """Generate suggestions for best-in-category based on review mentions."""
    if not scraped_data or not scraped_data.get('reviews'):
        return {
            'name': 'No review data available',
            'reason': 'Unable to find competitor recommendations without review analysis.',
            'rating': 'N/A'
        }
    
    return find_best_in_category_from_reviews(query, scraped_data['reviews'])

def get_category_suggestion(category: str) -> dict:
    """Return message indicating suggestions require specific product analysis."""
    return {
        'name': 'Specific Product Analysis Required',
        'reason': 'Please analyze a specific product to get recommendations based on actual user reviews.',
        'rating': 'N/A'
    }