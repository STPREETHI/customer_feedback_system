# Requirements.txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
nltk>=3.8
textblob>=0.17.1

# Setup Instructions

## Installation

1. Create a virtual environment:
```bash
python -m venv feedback_system
source feedback_system/bin/activate  # On Windows: feedback_system\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create the project structure:
```
feedback_system/
├── app.py                    # Main Streamlit application
├── preprocessing.py          # Text preprocessing module
├── nlp_models.py            # NLP analysis models
├── embeddings.py            # Embeddings and similarity search
├── rag_summarization.py     # RAG-based summarization
├── requirements.txt         # Dependencies
├── data/                    # Data directory
│   └── sample_feedback.csv  # Sample dataset
└── README.md               # Documentation
```

## Running the Application

1. Navigate to the project directory:
```bash
cd feedback_system
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser to http://localhost:8501

## Features

### 🎯 Main Features
- **Real-time Feedback Analysis**: Process customer feedback with AI models
- **Multi-dimensional Analysis**: Sentiment, topics, aspects, and emotions
- **Interactive Dashboard**: Modern, responsive UI with visualizations
- **Actionable Insights**: AI-generated recommendations and summaries
- **Semantic Search**: Find similar feedback using embeddings
- **Theme Detection**: Automatically identify emerging patterns

### 📊 Analytics Capabilities
- Sentiment trend analysis over time
- Topic distribution and popularity
- Aspect-based sentiment analysis
- Cluster analysis for theme identification
- Similarity search for related feedback

### 🤖 AI Components
- **Sentiment Analysis**: Lexicon-based with confidence scoring
- **Topic Classification**: Multi-label keyword-based classification
- **Aspect Analysis**: Detect sentiment for specific product aspects
- **Embedding Generation**: TF-IDF based text embeddings
- **RAG Summarization**: Template-based insight generation

### 📋 Data Processing
- Comprehensive text preprocessing
- Language detection
- Typo correction
- Special character handling
- Batch processing capabilities

## Customization

### Adding New Topics
Edit the `topic_keywords` dictionary in `nlp_models.py`:
```python
self.topic_keywords = {
    'new_topic': {
        'keyword1', 'keyword2', 'keyword3'
    }
}
```

### Modifying Sentiment Analysis
Update word lists in `SentimentAnalyzer` class:
```python
self.positive_words.update(['new_positive_word'])
self.negative_words.update(['new_negative_word'])
```

### Custom Data Sources
Upload CSV files with these required columns:
- `feedback`: The feedback text
- `timestamp`: Date/time of feedback
- `source`: Source of feedback (optional)
- `customer_id`: Customer identifier (optional)

## Production Deployment

### Recommended Upgrades for Production:

1. **Replace Mock Models**:
   - Use `transformers` library with `distilbert-base-uncased`
   - Implement `sentence-transformers` for embeddings
   - Add `flan-t5-small` for text generation
   - Use real FAISS for vector search

2. **Database Integration**:
   - PostgreSQL or MongoDB for feedback storage
   - Redis for caching embeddings
   - Vector databases like Pinecone or Weaviate

3. **Scalability**:
   - Docker containerization
   - Kubernetes deployment
   - Load balancing for multiple instances
   - Background task processing with Celery

4. **Security**:
   - User authentication
   - API key management
   - Data encryption
   - Rate limiting

5. **Monitoring**:
   - Application performance monitoring
   - Model performance tracking
   - Error logging and alerting
   - Usage analytics

## Sample Data Format

```csv
id,timestamp,feedback,source,customer_id
1,2024-01-15,"App crashes frequently",App Store,CUST001
2,2024-01-16,"Love the new dark mode!",In-App,CUST002
3,2024-01-17,"Support response was slow",Email,CUST003
```

## API Integration

### REST API Endpoints (Future Enhancement):
```
POST /api/feedback/analyze
GET /api/feedback/trends
GET /api/feedback/summary
POST /api/feedback/search
```

## Performance Optimization

- Implement caching for processed results
- Use batch processing for large datasets
- Optimize database queries
- Implement async processing for real-time analysis

## Troubleshooting

### Common Issues:

1. **ImportError**: Install missing dependencies with pip
2. **Memory Issues**: Reduce batch size for large datasets
3. **Slow Performance**: Enable caching and use smaller models
4. **Display Issues**: Clear browser cache and restart Streamlit

### Support:
For issues or questions, check the documentation or create an issue in the project repository.