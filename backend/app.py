from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
from functools import wraps
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Import enhanced modules ---
try:
    from scraper import get_general_reviews
    from rag import analyze_reviews_definitively, summarize_and_analyze_comparison
    logger.info("All modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    raise

app = Flask(__name__)
CORS(app)

# Request timing decorator
def log_timing(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{f.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{f.__name__} failed after {duration:.2f}s: {e}")
            raise
    return decorated_function

def validate_input(data, required_fields):
    """Validate request data"""
    if not data:
        return False, "No data provided"
    
    for field in required_fields:
        if not data.get(field) or not data.get(field).strip():
            return False, f"Field '{field}' is required and cannot be empty"
    
    return True, None

def create_error_response(message: str, status_code: int = 400, suggestion: str = None):
    """Create standardized error response"""
    response = {"error": message}
    if suggestion:
        response["suggestion"] = suggestion
    return jsonify(response), status_code

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0"
    })

@app.route('/api/analyze_product', methods=['POST'])
@log_timing
def analyze_product():
    """Enhanced product analysis endpoint with better error handling"""
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_msg = validate_input(data, ['query'])
        if not is_valid:
            return create_error_response(
                error_msg, 
                400, 
                "Please provide a valid product name (e.g., 'iPhone 15 Pro') or URL"
            )
        
        query = data['query'].strip()
        
        # Additional validation
        if len(query) < 3:
            return create_error_response(
                "Query too short",
                400,
                "Please enter at least 3 characters for the product name"
            )
        
        if len(query) > 200:
            return create_error_response(
                "Query too long", 
                400,
                "Please limit your search to 200 characters or less"
            )
        
        logger.info(f"Starting analysis for: {query}")
        
        # Step 1: Get reviews
        scrape_result = get_general_reviews(query)
        
        if scrape_result["status"] != "success":
            error_msg = scrape_result.get("message", "Failed to retrieve reviews")
            suggestions = {
                "no_data": "Try a more specific product name or check the spelling",
                "timeout": "The request timed out. Please try again",
                "blocked": "Too many requests. Please wait a moment and try again"
            }
            suggestion = suggestions.get(scrape_result.get("error_type"), "Please try a different product or check your connection")
            
            return create_error_response(error_msg, 404, suggestion)
        
        logger.info(f"Retrieved {len(scrape_result.get('reviews', []))} reviews from {len(scrape_result.get('sources', []))} sources")
        
        # Step 2: Analyze reviews
        analysis = analyze_reviews_definitively(scrape_result["reviews"])
        
        # Add metadata
        analysis["product_name"] = scrape_result["product_name"]
        analysis["sources"] = scrape_result.get("sources", [])
        analysis["review_count"] = scrape_result.get("review_count", len(scrape_result["reviews"]))
        analysis["analysis_timestamp"] = time.time()
        
        logger.info(f"Analysis completed for: {analysis['product_name']}")
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Analyze product error: {str(e)}\n{traceback.format_exc()}")
        
        # Specific error handling
        if "timeout" in str(e).lower():
            return create_error_response(
                "Request timed out while analyzing the product",
                408,
                "Please try again. If the problem persists, try a more specific product name"
            )
        elif "connection" in str(e).lower():
            return create_error_response(
                "Connection error occurred",
                503,
                "Please check your internet connection and try again"
            )
        elif "memory" in str(e).lower():
            return create_error_response(
                "Server overloaded",
                503,
                "The server is currently busy. Please try again in a few minutes"
            )
        else:
            return create_error_response(
                "An unexpected error occurred during analysis",
                500,
                "Please try again or contact support if the issue persists"
            )

@app.route('/api/compare_products', methods=['POST'])
@log_timing
def compare_products():
    """Enhanced product comparison endpoint"""
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_msg = validate_input(data, ['query1', 'query2'])
        if not is_valid:
            return create_error_response(
                error_msg,
                400,
                "Please provide two valid product names to compare"
            )
        
        query1 = data['query1'].strip()
        query2 = data['query2'].strip()
        
        # Additional validation
        for i, query in enumerate([query1, query2], 1):
            if len(query) < 3:
                return create_error_response(
                    f"Product {i} name too short",
                    400,
                    "Each product name must be at least 3 characters long"
                )
            if len(query) > 200:
                return create_error_response(
                    f"Product {i} name too long",
                    400,
                    "Each product name must be 200 characters or less"
                )
        
        # Check if queries are too similar
        if query1.lower() == query2.lower():
            return create_error_response(
                "Cannot compare identical products",
                400,
                "Please provide two different product names for comparison"
            )
        
        logger.info(f"Starting comparison: '{query1}' vs '{query2}'")
        
        # Analyze first product
        logger.info("Analyzing first product...")
        scrape_result1 = get_general_reviews(query1)
        if scrape_result1["status"] != "success":
            return create_error_response(
                f"Could not analyze '{query1}': {scrape_result1['message']}",
                404,
                "Please check the first product name and try again"
            )
        
        analysis1 = analyze_reviews_definitively(scrape_result1["reviews"])
        analysis1["product_name"] = scrape_result1["product_name"]
        analysis1["sources"] = scrape_result1.get("sources", [])
        analysis1["review_count"] = scrape_result1.get("review_count", len(scrape_result1["reviews"]))
        
        logger.info(f"First product analysis complete: {analysis1['product_name']}")
        
        # Analyze second product  
        logger.info("Analyzing second product...")
        scrape_result2 = get_general_reviews(query2)
        if scrape_result2["status"] != "success":
            return create_error_response(
                f"Could not analyze '{query2}': {scrape_result2['message']}",
                404,
                "Please check the second product name and try again"
            )
        
        analysis2 = analyze_reviews_definitively(scrape_result2["reviews"])
        analysis2["product_name"] = scrape_result2["product_name"]
        analysis2["sources"] = scrape_result2.get("sources", [])
        analysis2["review_count"] = scrape_result2.get("review_count", len(scrape_result2["reviews"]))
        
        logger.info(f"Second product analysis complete: {analysis2['product_name']}")
        
        # Generate comparison summary
        logger.info("Generating comparison summary...")
        comparison_summary = summarize_and_analyze_comparison(analysis1, analysis2)
        
        result = {
            "comparison_summary": comparison_summary,
            "product1": analysis1,
            "product2": analysis2,
            "comparison_timestamp": time.time()
        }
        
        logger.info("Comparison completed successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Compare products error: {str(e)}\n{traceback.format_exc()}")
        
        if "timeout" in str(e).lower():
            return create_error_response(
                "Comparison timed out",
                408,
                "The comparison is taking too long. Please try with more specific product names"
            )
        elif "memory" in str(e).lower():
            return create_error_response(
                "Server overloaded during comparison",
                503,
                "Please try again in a few minutes"
            )
        else:
            return create_error_response(
                "An error occurred during product comparison",
                500,
                "Please try again or use more specific product names"
            )

@app.errorhandler(404)
def not_found(error):
    return create_error_response(
        "API endpoint not found",
        404,
        "Please check the URL and try again"
    )

@app.errorhandler(405)
def method_not_allowed(error):
    return create_error_response(
        "Method not allowed",
        405,
        "Please use POST method for this endpoint"
    )

@app.errorhandler(413)
def request_entity_too_large(error):
    return create_error_response(
        "Request too large",
        413,
        "Please reduce the size of your request"
    )

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return create_error_response(
        "Too many requests",
        429,
        "Please wait a moment before making another request"
    )

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    return create_error_response(
        "Internal server error",
        500,
        "Please try again later or contact support"
    )

if __name__ == '__main__':
    logger.info("Starting Universal Product Intelligence Engine API")
    logger.info("Server will be available at http://127.0.0.1:5000")
    app.run(debug=False, host='127.0.0.1', port=5000)