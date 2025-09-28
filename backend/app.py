from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging

# --- Import the optimized functions ---
from scraper import get_general_reviews
from rag import analyze_reviews_definitively, summarize_and_analyze_comparison, detect_product_category, get_category_suggestion

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze_product', methods=['POST'])
def analyze_product():
    """ The optimized endpoint for analyzing any product name or URL. """
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
    log.info(f"Analysis completed for: {analysis['product_name']} in {end_time - start_time:.2f}s")
    
    return jsonify(analysis)


@app.route('/api/compare_products', methods=['POST'])
def compare_products():
    """ Ultra-fast comparison endpoint with parallel processing. """
    start_time = time.time()
    data = request.get_json()
    query1, query2 = data.get('query1'), data.get('query2')
    log.info(f"Received comparison request for: '{query1}' vs '{query2}'")

    if not query1 or not query2:
        return jsonify({"error": "Two product names or URLs are required."}), 400

    try:
        # Parallel scraping and analysis for speed
        import concurrent.futures
        import threading
        
        def analyze_single_product(query):
            scrape_result = get_general_reviews(query, num_results=3)  # Reduced results for speed
            if scrape_result["status"] != "success":
                return {"error": scrape_result["message"]}
            
            # Ultra-fast analysis with even more aggressive limits
            reviews = scrape_result["reviews"][:15]  # Max 15 reviews for comparison
            analysis = analyze_reviews_definitively(reviews)
            analysis["product_name"] = scrape_result["product_name"]
            return analysis

        # Run both analyses in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(analyze_single_product, query1)
            future2 = executor.submit(analyze_single_product, query2)
            
            analysis1 = future1.result(timeout=40)  # 40s max per product
            analysis2 = future2.result(timeout=40)

        # Check for errors
        if "error" in analysis1:
            return jsonify({"error": f"Could not analyze '{query1}': {analysis1['error']}"}), 404
        if "error" in analysis2:
            return jsonify({"error": f"Could not analyze '{query2}': {analysis2['error']}"}), 404
        
        # Quick comparison generation
        comparison_result = summarize_and_analyze_comparison(analysis1, analysis2)

        end_time = time.time()
        log.info(f"compare_products completed in {end_time - start_time:.2f}s")

        return jsonify({
            "comparison_summary": comparison_result["comparison_summary"],
            "recommendation": comparison_result["recommendation"],
            "product1": analysis1,
            "product2": analysis2
        })
        
    except concurrent.futures.TimeoutError:
        return jsonify({"error": "Comparison timed out. Please try with simpler product names."}), 408
    except Exception as e:
        log.error(f"Comparison failed: {str(e)}")
        return jsonify({"error": "Comparison failed. Please try again."}), 500

@app.route('/api/suggest_best_product', methods=['POST'])
def suggest_best_product():
    """ New endpoint for getting category-based product suggestions. """
    data = request.get_json()
    category = data.get('category', 'general')
    
    log.info(f"Received suggestion request for category: {category}")
    
    suggestion = get_category_suggestion(category)
    
    return jsonify({
        "status": "success",
        "category": category,
        "suggestion": suggestion
    })

@app.route('/api/detect_category', methods=['POST'])
def detect_category():
    """ Endpoint to detect product category from name. """
    data = request.get_json()
    product_name = data.get('product_name', '')
    
    if not product_name:
        return jsonify({"error": "Product name is required."}), 400
    
    category = detect_product_category(product_name)
    suggestion = get_category_suggestion(category)
    
    return jsonify({
        "status": "success",
        "product_name": product_name,
        "detected_category": category,
        "suggestion": suggestion
    })

if __name__ == '__main__':
    app.run(debug=False)