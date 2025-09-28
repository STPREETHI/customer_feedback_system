from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging

# --- Import the final, correct function names ---
from scraper import get_general_reviews
from rag import analyze_reviews_definitively, summarize_and_analyze_comparison

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze_product', methods=['POST'])
def analyze_product():
    """ The definitive endpoint for analyzing any product name or URL. """
    start_time = time.time()
    data = request.get_json()
    query = data.get('query')
    log.info(f"Received analysis request for: {query}")

    if not query:
        return jsonify({"error": "A product name or URL is required."}), 400

    scrape_result = get_general_reviews(query)
    if scrape_result["status"] != "success":
        log.error(f"Scraping failed: {scrape_result['message']}")
        return jsonify({"error": scrape_result["message"]}), 404

    analysis = analyze_reviews_definitively(scrape_result["reviews"])
    analysis["product_name"] = scrape_result["product_name"]
    analysis["sources"] = scrape_result.get("sources", [])
    
    end_time = time.time()
    log.info(f"Analysis completed for: {analysis['product_name']}")
    log.info(f"analyze_product completed in {end_time - start_time:.2f}s")
    
    return jsonify(analysis)


@app.route('/api/compare_products', methods=['POST'])
def compare_products():
    """ The definitive endpoint for comparing any two product names or URLs. """
    start_time = time.time()
    data = request.get_json()
    query1, query2 = data.get('query1'), data.get('query2')
    log.info(f"Received comparison request for: '{query1}' vs '{query2}'")


    if not query1 or not query2:
        return jsonify({"error": "Two product names or URLs are required."}), 400

    # Analyze the first product
    scrape_result1 = get_general_reviews(query1)
    if scrape_result1["status"] != "success":
        return jsonify({"error": f"Could not analyze '{query1}': {scrape_result1['message']}"}), 404
    analysis1 = analyze_reviews_definitively(scrape_result1["reviews"])
    analysis1["product_name"] = scrape_result1["product_name"]

    # Analyze the second product
    scrape_result2 = get_general_reviews(query2)
    if scrape_result2["status"] != "success":
        return jsonify({"error": f"Could not analyze '{query2}': {scrape_result2['message']}"}), 404
    analysis2 = analyze_reviews_definitively(scrape_result2["reviews"])
    analysis2["product_name"] = scrape_result2["product_name"]
    
    # Generate the detailed AI comparison
    comparison_summary = summarize_and_analyze_comparison(analysis1, analysis2)

    end_time = time.time()
    log.info(f"compare_products completed in {end_time - start_time:.2f}s")

    return jsonify({
        "comparison_summary": comparison_summary,
        "product1": analysis1,
        "product2": analysis2
    })

if __name__ == '__main__':
    app.run(debug=False)

