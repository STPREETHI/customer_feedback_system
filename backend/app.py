from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import re

# Import from our backend modules
from database import get_db_connection
from rag import generate_summary, answer_query_with_rag, generate_pros_and_cons, generate_text
from scraper import scrape_reviews_from_url
from preprocessing import clean_text
from models import predict_sentiment

# Initialize Flask App
app = Flask(__name__)
CORS(app) 

# --- Pre-load models on startup ---
print("--- Initializing server: Loading all AI models... ---")
try:
    answer_query_with_rag("preload_models") 
    print("--- Models loaded successfully. Server is ready. ---")
except Exception as e:
    print(f"--- CRITICAL: Models failed to load on startup: {e} ---")

def analyze_scraped_reviews(reviews):
    """Helper function to run analysis on a list of scraped review texts."""
    if not reviews:
        return None
    cleaned_reviews = [clean_text(r) for r in reviews]
    sentiment_results = predict_sentiment(cleaned_reviews)

    df = pd.DataFrame(sentiment_results)
    df['sentiment'] = df['label']
    df['sentiment_score'] = np.where(df['label'] == 'POSITIVE', df['score'], 1 - df['score'])

    overall_score = df['sentiment_score'].mean()
    verdict = "Good Pick!" if overall_score > 0.6 else "Could Be Better"
    
    summary_context = " ".join(cleaned_reviews)
    ai_summary = generate_summary(f"Summarize the key points from these customer reviews: {summary_context}")

    return {
        "verdict": verdict,
        "overall_score": float(overall_score),
        "ai_summary": ai_summary,
        "total_reviews_analyzed": len(reviews)
    }

# --- API Routes ---

@app.route('/api/analyze_url', methods=['POST'])
def analyze_url():
    """Analyzes reviews from a single live product URL."""
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "URL is required."}), 400

    try:
        reviews = scrape_reviews_from_url(url)
        if not reviews:
            return jsonify({"error": "Could not fetch reviews for this URL. The page might be protected or the layout has changed."}), 404
        
        analysis_result = analyze_scraped_reviews(reviews)
        return jsonify(analysis_result)

    except Exception as e:
        print(f"Error in /api/analyze_url: {e}")
        return jsonify({"error": "Failed to analyze the provided URL."}), 500

@app.route('/api/compare_urls', methods=['POST'])
def compare_urls():
    """NEW: Scrapes and compares two live product URLs."""
    data = request.get_json()
    url1, url2 = data.get('url1'), data.get('url2')
    if not url1 or not url2:
        return jsonify({"error": "Two URLs are required for comparison."}), 400
    
    try:
        reviews1 = scrape_reviews_from_url(url1)
        reviews2 = scrape_reviews_from_url(url2)
        
        details1 = analyze_scraped_reviews(reviews1)
        details2 = analyze_scraped_reviews(reviews2)

        if not details1 or not details2:
            return jsonify({"error": "Could not analyze one or both of the products."}), 500

        # Generate a high-level comparison and recommendation
        prompt = (
            "You are a product recommendation expert. Based on the two product summaries below, "
            "provide a final recommendation on which one is the better choice and a brief summary of why. "
            "Start your response with 'Recommendation:' followed by the better product. \n\n"
            f"Product 1 (Score: {details1['overall_score']:.2f}): {details1['ai_summary']}\n\n"
            f"Product 2 (Score: {details2['overall_score']:.2f}): {details2['ai_summary']}\n\n"
            "Final Recommendation and reasoning:"
        )
        
        full_comparison_text = generate_text(prompt, max_length=300)
        
        # Split recommendation from the main summary
        recommendation = "Product 1 is the better choice." if details1['overall_score'] > details2['overall_score'] else "Product 2 is the better choice."
        comparison_summary = full_comparison_text

        # A simple regex to find the first sentence for the main recommendation
        match = re.search(r"Recommendation:.*?\.", full_comparison_text)
        if match:
            recommendation = match.group(0)
            comparison_summary = full_comparison_text.replace(recommendation, "").strip()


        return jsonify({
            "product1_details": details1,
            "product2_details": details2,
            "recommendation": recommendation,
            "comparison_summary": comparison_summary,
        })

    except Exception as e:
        print(f"Error in /api/compare_urls: {e}")
        return jsonify({"error": "An error occurred during live comparison."}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Endpoint for the Suggestion Bot. Now detects URLs to provide pros/cons.
    """
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "A query is required."}), 400

    # Regex to find a URL in the user's query
    url_match = re.search(r'https?://[^\s]+', query)

    try:
        if url_match:
            # If a URL is found, scrape it and generate pros and cons
            url = url_match.group(0)
            reviews = scrape_reviews_from_url(url)
            if not reviews:
                answer = "I couldn't fetch reviews from that URL. It might be protected or an unsupported format."
            else:
                cleaned_reviews_text = " ".join([clean_text(r) for r in reviews])
                answer = generate_pros_and_cons(cleaned_reviews_text)
        else:
            # Otherwise, use the existing RAG pipeline on the local dataset
            answer = answer_query_with_rag(query)
            
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error in /api/recommend: {e}")
        return jsonify({"error": "An error occurred while generating the recommendation."}), 500

# --- Existing API Routes (No changes needed below) ---
@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT DISTINCT product_id FROM feedback ORDER BY product_id", conn)
        conn.close()
        return jsonify(df['product_id'].tolist())
    except Exception as e:
        print(f"Error in /api/products: {e}")
        return jsonify({"error": "Failed to fetch product list."}), 500

@app.route('/api/product_details/<product_id>', methods=['GET'])
def get_product_details(product_id):
    try:
        conn = get_db_connection()
        query = "SELECT sentiment, sentiment_score, cleaned_text FROM feedback WHERE product_id = ?"
        df = pd.read_sql_query(query, conn, params=(product_id,))
        conn.close()

        if df.empty: return jsonify({"error": "Product not found"}), 404
        
        sentiment_distribution = {k: int(v) for k, v in df['sentiment'].value_counts().to_dict().items()}
        overall_score = df[df['sentiment'] == 'POSITIVE']['sentiment_score'].mean() - df[df['sentiment'] == 'NEGATIVE']['sentiment_score'].mean()
        verdict = "Overall Good" if overall_score > 0 else "Overall Bad"
        sample_reviews = " ".join(df['cleaned_text'].sample(n=min(len(df), 25), random_state=42).tolist())
        ai_summary = generate_summary(sample_reviews)

        return jsonify({
            "product_id": product_id, "total_reviews": len(df),
            "sentiment_distribution": sentiment_distribution,
            "overall_score": float(overall_score), "verdict": verdict, "ai_summary": ai_summary
        })
    except Exception as e:
        print(f"Error in /api/product_details/{product_id}: {e}")
        return jsonify({"error": "Failed to fetch product details."}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)

