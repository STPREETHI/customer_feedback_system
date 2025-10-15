from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
import spacy
from spacy.cli import download

# --- Import your project's functions ---
from scraper import get_general_reviews, create_driver 
from rag import analyze_reviews_definitively, summarize_and_analyze_comparison, detect_product_category, get_category_suggestion, generate_context_based_suggestions

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- Load the spaCy NLP model ---
try:
    log.info("Loading spaCy English model...")
    nlp = spacy.load("en_core_web_sm")
    log.info("spaCy model loaded successfully.")
except OSError:
    log.warning("spaCy model not found. Downloading now...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Helper function to classify sentiment for individual reviews ---
def classify_individual_sentiments(reviews):
    classified = []
    positive_words = ["good", "excellent", "amazing", "love", "great", "best", "fast", "beautiful", "impressive", "solid"]
    negative_words = ["bad", "poor", "disappointed", "problem", "slow", "expensive", "lacks", "only"]
    for review in reviews:
        review_lower = review.lower()
        sentiment = "Neutral"
        has_pos = any(word in review_lower for word in positive_words)
        has_neg = any(word in review_lower for word in negative_words)
        
        if has_pos and not has_neg:
            sentiment = "Positive"
        elif has_neg and not has_pos:
            sentiment = "Negative"
        
        classified.append({"text": review, "sentiment": sentiment})
    return classified

@app.route('/api/analyze_product', methods=['POST'])
def analyze_product():
    """Endpoint for analyzing a single product."""
    start_time = time.time()
    data = request.get_json()
    query = data.get('query')
    if not query: return jsonify({"error": "A product name or URL is required."}), 400

    scrape_result = get_general_reviews(query) 
    if scrape_result["status"] != "success":
        return jsonify({"error": scrape_result["message"]}), 404

    analysis = analyze_reviews_definitively(scrape_result["reviews"])
    analysis["product_name"] = scrape_result["product_name"]
    analysis["sources"] = scrape_result.get("sources", [])
    analysis["reviews"] = scrape_result.get("reviews", [])
    
    # --- ADDITION: Add classified reviews to the response ---
    analysis["classified_reviews"] = classify_individual_sentiments(analysis["reviews"])
    
    log.info(f"Analysis completed for '{analysis['product_name']}' in {time.time() - start_time:.2f}s")
    return jsonify(analysis)

@app.route('/api/compare_products', methods=['POST'])
def compare_products():
    """Endpoint for comparing two products."""
    start_time = time.time()
    data = request.get_json()
    query1, query2 = data.get('query1'), data.get('query2')
    if not query1 or not query2: return jsonify({"error": "Two product names are required."}), 400

    try:
        import concurrent.futures
        def analyze_single_product_safely(query):
            temp_driver = create_driver()
            if not temp_driver: return {"error": "Could not create a browser for analysis."}
            try:
                scrape_result = get_general_reviews(query, num_results=3, driver=temp_driver)
                if scrape_result["status"] != "success": return {"error": scrape_result["message"]}
                
                analysis = analyze_reviews_definitively(scrape_result["reviews"][:15])
                analysis["product_name"] = scrape_result["product_name"]
                analysis["reviews"] = scrape_result.get("reviews", [])
                
                # --- ADDITION: Add classified reviews for each product in comparison ---
                analysis["classified_reviews"] = classify_individual_sentiments(analysis["reviews"])
                
                return analysis
            finally:
                log.info(f"Closing temporary browser for '{query}' analysis.")
                temp_driver.quit()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(analyze_single_product_safely, query1)
            future2 = executor.submit(analyze_single_product_safely, query2)
            analysis1 = future1.result(timeout=90) 
            analysis2 = future2.result(timeout=90)

        if "error" in analysis1: return jsonify({"error": f"Could not analyze '{query1}': {analysis1['error']}"}), 404
        if "error" in analysis2: return jsonify({"error": f"Could not analyze '{query2}': {analysis2['error']}"}), 404
        
        comparison_result = summarize_and_analyze_comparison(analysis1, analysis2)
        log.info(f"Comparison completed in {time.time() - start_time:.2f}s")

        return jsonify({
            "comparison_summary": comparison_result["comparison_summary"],
            "recommendation": comparison_result["recommendation"],
            "product1": analysis1,
            "product2": analysis2
        })
    except Exception as e:
        log.error(f"A critical error occurred during comparison: {str(e)}")
        return jsonify({"error": "A critical error occurred during comparison."}), 500

# (Other endpoints remain unchanged)
@app.route('/api/suggest_best_product', methods=['POST'])
def suggest_best_product():
    data = request.get_json()
    query = data.get('query', '')
    if query and query.strip():
        scrape_result = get_general_reviews(query, num_results=2)
        if scrape_result["status"] == "success":
            suggestion = generate_context_based_suggestions(query, scrape_result)
        else:
            suggestion = {'name': 'No Review Data', 'reason': 'Could not find data for recommendations.', 'rating': 'N/A'}
    else:
        suggestion = {'name': 'Specific Product Required', 'reason': 'Analyze a product to get recommendations.', 'rating': 'N/A'}
    return jsonify({"status": "success", "suggestion": suggestion})

@app.route('/api/detect_category', methods=['POST'])
def detect_category():
    data = request.get_json()
    product_name = data.get('product_name', '')
    if not product_name: return jsonify({"error": "Product name is required."}), 400
    category = detect_product_category(product_name)
    suggestion = get_category_suggestion(category)
    return jsonify({"status": "success", "product_name": product_name, "detected_category": category, "suggestion": suggestion})

@app.route('/api/parse_sentence', methods=['POST'])
def parse_sentence():
    data = request.get_json()
    sentence = data.get('sentence', '')
    if not sentence: return jsonify({"error": "A sentence is required."}), 400
    try:
        doc = nlp(sentence)
        def token_to_dict(token):
            children = [token_to_dict(child) for child in token.children]
            return {"label": f"{token.pos_} ({token.dep_})", "word": token.text, "children": children}
        root_token = next((token for token in doc if token.head == token), doc[0] if len(doc) > 0 else None)
        if not root_token: return jsonify({"error": "Could not process the sentence."}), 400
        tree_dict = token_to_dict(root_token)
        return jsonify({"parsed_tree": tree_dict})
    except Exception as e:
        log.error(f"spaCy parsing failed: {str(e)}")
        return jsonify({"error": "An internal error occurred during parsing."}), 500

if __name__ == '__main__':
    app.run(debug=True)