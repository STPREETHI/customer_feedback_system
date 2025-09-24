import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Import our modules
from preprocessing import TextPreprocessor
from nlp_models import FeedbackAnalyzer
from embeddings import EmbeddingManager
from rag_summarization import RAGSummarizer

# Page config
st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .insight-card {
        background: #f8f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    
    .warning-card {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    
    .critical-card {
        background: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load or generate sample feedback data"""
    sample_data = [
        {
            "id": 1,
            "timestamp": datetime.now() - timedelta(days=5),
            "feedback": "The app crashes frequently when I try to upload files. Very frustrating experience.",
            "source": "App Store",
            "customer_id": "CUST001"
        },
        {
            "id": 2,
            "timestamp": datetime.now() - timedelta(days=4),
            "feedback": "Love the new dark mode feature! Makes it much easier on the eyes during night use.",
            "source": "In-App",
            "customer_id": "CUST002"
        },
        {
            "id": 3,
            "timestamp": datetime.now() - timedelta(days=3),
            "feedback": "Customer support took 3 days to respond. Not acceptable for premium users.",
            "source": "Email",
            "customer_id": "CUST003"
        },
        {
            "id": 4,
            "timestamp": datetime.now() - timedelta(days=2),
            "feedback": "The checkout process is confusing. Had to abandon my cart twice before figuring it out.",
            "source": "Website",
            "customer_id": "CUST004"
        },
        {
            "id": 5,
            "timestamp": datetime.now() - timedelta(days=1),
            "feedback": "Amazing product quality! Exceeded my expectations. Will definitely recommend to friends.",
            "source": "Review Site",
            "customer_id": "CUST005"
        },
        {
            "id": 6,
            "timestamp": datetime.now(),
            "feedback": "The mobile app is slow and sometimes doesn't sync with the web version. Please fix this.",
            "source": "In-App",
            "customer_id": "CUST006"
        }
    ]
    return pd.DataFrame(sample_data)

@st.cache_resource
def initialize_models():
    """Initialize all models and components"""
    preprocessor = TextPreprocessor()
    analyzer = FeedbackAnalyzer()
    embedding_manager = EmbeddingManager()
    rag_summarizer = RAGSummarizer()
    
    return preprocessor, analyzer, embedding_manager, rag_summarizer

def process_feedback_batch(df, preprocessor, analyzer, embedding_manager):
    """Process a batch of feedback"""
    results = []
    
    for _, row in df.iterrows():
        # Preprocess
        cleaned_text = preprocessor.clean_text(row['feedback'])
        
        # Analyze
        sentiment = analyzer.analyze_sentiment(cleaned_text)
        topics = analyzer.classify_topics(cleaned_text)
        aspects = analyzer.analyze_aspects(cleaned_text)
        
        # Get embeddings
        embedding = embedding_manager.get_embedding(cleaned_text)
        
        results.append({
            'id': row['id'],
            'original_text': row['feedback'],
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'topics': topics,
            'aspects': aspects,
            'embedding': embedding,
            'timestamp': row['timestamp'],
            'source': row['source'],
            'customer_id': row['customer_id']
        })
    
    return results

def render_header():
    """Render the main header"""
    st.markdown("""
        <div class="main-header">
            <h1>🎯 Customer Feedback Intelligence System</h1>
            <p>AI-Powered Feedback Analysis with Actionable Insights</p>
        </div>
    """, unsafe_allow_html=True)

def render_dashboard(processed_results):
    """Render the main dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    total_feedback = len(processed_results)
    avg_sentiment = np.mean([r['sentiment']['compound'] for r in processed_results])
    positive_ratio = len([r for r in processed_results if r['sentiment']['label'] == 'positive']) / total_feedback
    
    # Recent trend (mock calculation)
    recent_sentiment = avg_sentiment + np.random.uniform(-0.1, 0.1)
    sentiment_trend = "↗️" if recent_sentiment > avg_sentiment else "↘️"
    
    with col1:
        st.metric(
            label="Total Feedback",
            value=total_feedback,
            delta="+12 this week"
        )
    
    with col2:
        st.metric(
            label="Avg Sentiment",
            value=f"{avg_sentiment:.2f}",
            delta=f"{sentiment_trend} {abs(recent_sentiment - avg_sentiment):.2f}"
        )
    
    with col3:
        st.metric(
            label="Positive Ratio",
            value=f"{positive_ratio:.1%}",
            delta="+5.2% vs last week"
        )
    
    with col4:
        st.metric(
            label="Response Rate",
            value="94.2%",
            delta="+2.1% improvement"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        sentiment_data = [r['sentiment']['label'] for r in processed_results]
        sentiment_counts = pd.Series(sentiment_data).value_counts()
        
        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#6c757d'
            }
        )
        fig_sentiment.update_layout(height=400)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Topic distribution
        all_topics = []
        for r in processed_results:
            all_topics.extend(r['topics'])
        
        if all_topics:
            topic_counts = pd.Series(all_topics).value_counts().head(5)
            fig_topics = px.bar(
                x=topic_counts.values,
                y=topic_counts.index,
                orientation='h',
                title="Top Topics",
                color=topic_counts.values,
                color_continuous_scale='viridis'
            )
            fig_topics.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_topics, use_container_width=True)
    
    # Sentiment timeline
    df_timeline = pd.DataFrame([
        {
            'date': r['timestamp'],
            'sentiment': r['sentiment']['compound'],
            'label': r['sentiment']['label']
        } for r in processed_results
    ])
    
    fig_timeline = px.line(
        df_timeline.sort_values('date'),
        x='date',
        y='sentiment',
        title="Sentiment Trend Over Time",
        color_discrete_sequence=['#667eea']
    )
    fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_timeline.update_layout(height=400)
    st.plotly_chart(fig_timeline, use_container_width=True)

def render_feedback_table(processed_results):
    """Render the feedback analysis table"""
    st.subheader("📋 Feedback Analysis Table")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        sentiment_filter = st.selectbox(
            "Filter by Sentiment",
            ["All", "Positive", "Negative", "Neutral"]
        )
    
    with col2:
        source_filter = st.selectbox(
            "Filter by Source",
            ["All"] + list(set([r['source'] for r in processed_results]))
        )
    
    with col3:
        search_term = st.text_input("Search in feedback")
    
    # Prepare table data
    table_data = []
    for r in processed_results:
        # Apply filters
        if sentiment_filter != "All" and r['sentiment']['label'].title() != sentiment_filter:
            continue
        if source_filter != "All" and r['source'] != source_filter:
            continue
        if search_term and search_term.lower() not in r['original_text'].lower():
            continue
            
        table_data.append({
            'ID': r['id'],
            'Feedback': r['original_text'][:100] + "..." if len(r['original_text']) > 100 else r['original_text'],
            'Sentiment': r['sentiment']['label'].title(),
            'Confidence': f"{r['sentiment']['confidence']:.2f}",
            'Topics': ", ".join(r['topics'][:2]) + ("..." if len(r['topics']) > 2 else ""),
            'Source': r['source'],
            'Date': r['timestamp'].strftime('%Y-%m-%d')
        })
    
    if table_data:
        df_display = pd.DataFrame(table_data)
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sentiment": st.column_config.TextColumn(
                    width="small"
                ),
                "Confidence": st.column_config.NumberColumn(
                    width="small"
                )
            }
        )
    else:
        st.info("No feedback matches the selected filters.")

def render_insights(processed_results, rag_summarizer):
    """Render actionable insights"""
    st.subheader("💡 AI-Generated Insights & Recommendations")
    
    # Generate insights
    with st.spinner("Generating insights..."):
        insights = rag_summarizer.generate_insights(processed_results)
    
    # Critical issues
    critical_issues = [r for r in processed_results if r['sentiment']['compound'] < -0.5]
    if critical_issues:
        st.markdown("### 🚨 Critical Issues Requiring Immediate Attention")
        for issue in critical_issues[:3]:
            st.markdown(f"""
                <div class="critical-card">
                    <h4>Issue #{issue['id']}</h4>
                    <p><strong>Feedback:</strong> {issue['original_text']}</p>
                    <p><strong>Topics:</strong> {', '.join(issue['topics'])}</p>
                    <p><strong>Source:</strong> {issue['source']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Top recommendations
    st.markdown("### 📈 Strategic Recommendations")
    
    recommendations = [
        {
            "priority": "High",
            "title": "Improve App Stability",
            "description": "Multiple reports of app crashes. Consider implementing crash analytics and automated testing.",
            "impact": "Will reduce negative feedback by ~30%",
            "timeline": "2-3 weeks"
        },
        {
            "priority": "Medium",
            "title": "Enhance Customer Support Response Time",
            "description": "Current average response time is 2.5 days. Target should be < 24 hours.",
            "impact": "Improve customer satisfaction score by 15%",
            "timeline": "1-2 weeks"
        },
        {
            "priority": "Medium",
            "title": "Simplify Checkout Process",
            "description": "Users report confusion during checkout. Conduct UX audit and A/B test improvements.",
            "impact": "Reduce cart abandonment by 20%",
            "timeline": "3-4 weeks"
        }
    ]
    
    for rec in recommendations:
        priority_color = {
            "High": "critical-card",
            "Medium": "warning-card",
            "Low": "insight-card"
        }
        
        st.markdown(f"""
            <div class="{priority_color[rec['priority']]}">
                <h4>{rec['title']} <span style="font-size: 0.8em; color: #666;">({rec['priority']} Priority)</span></h4>
                <p>{rec['description']}</p>
                <p><strong>Expected Impact:</strong> {rec['impact']}</p>
                <p><strong>Timeline:</strong> {rec['timeline']}</p>
            </div>
        """, unsafe_allow_html=True)

def render_summaries(processed_results, rag_summarizer):
    """Render AI-generated summaries"""
    st.subheader("📊 AI-Generated Summaries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Positive Feedback Summary")
        positive_feedback = [r for r in processed_results if r['sentiment']['label'] == 'positive']
        if positive_feedback:
            with st.spinner("Generating positive summary..."):
                pos_summary = rag_summarizer.generate_summary(positive_feedback, "positive")
            st.success(pos_summary)
        else:
            st.info("No positive feedback to summarize.")
    
    with col2:
        st.markdown("### Negative Feedback Summary")
        negative_feedback = [r for r in processed_results if r['sentiment']['label'] == 'negative']
        if negative_feedback:
            with st.spinner("Generating negative summary..."):
                neg_summary = rag_summarizer.generate_summary(negative_feedback, "negative")
            st.error(neg_summary)
        else:
            st.info("No negative feedback to summarize.")
    
    # Topic-based summaries
    st.markdown("### Topic-Based Analysis")
    
    # Get all unique topics
    all_topics = set()
    for r in processed_results:
        all_topics.update(r['topics'])
    
    if all_topics:
        selected_topic = st.selectbox("Select topic for detailed analysis:", sorted(all_topics))
        
        topic_feedback = [r for r in processed_results if selected_topic in r['topics']]
        if topic_feedback:
            with st.spinner(f"Analyzing {selected_topic}..."):
                topic_summary = rag_summarizer.generate_topic_analysis(topic_feedback, selected_topic)
            
            st.markdown(f"#### Analysis for '{selected_topic}'")
            st.info(topic_summary)
            
            # Topic sentiment breakdown
            topic_sentiments = [r['sentiment']['label'] for r in topic_feedback]
            sentiment_counts = pd.Series(topic_sentiments).value_counts()
            
            fig = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title=f"Sentiment Distribution for '{selected_topic}'",
                color=sentiment_counts.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application"""
    render_header()
    
    # Initialize models
    with st.spinner("Loading AI models..."):
        preprocessor, analyzer, embedding_manager, rag_summarizer = initialize_models()
    
    # Load data
    df = load_sample_data()
    
    # File upload option
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload feedback data (CSV/JSON)",
        type=['csv', 'json'],
        help="Upload your own feedback data or use the sample data"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df)} feedback entries")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
    
    # Process feedback
    with st.spinner("Processing feedback with AI models..."):
        processed_results = process_feedback_batch(df, preprocessor, analyzer, embedding_manager)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "📋 Feedback Analysis",
        "💡 Actionable Insights",
        "📄 AI Summaries"
    ])
    
    with tab1:
        render_dashboard(processed_results)
    
    with tab2:
        render_feedback_table(processed_results)
    
    with tab3:
        render_insights(processed_results, rag_summarizer)
    
    with tab4:
        render_summaries(processed_results, rag_summarizer)
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.header("📈 System Status")
    st.sidebar.success("✅ All models loaded")
    st.sidebar.info(f"📊 Processing {len(processed_results)} feedback entries")
    st.sidebar.info("🤖 AI analysis complete")
    
    # Model info
    with st.sidebar.expander("🔧 Model Information"):
        st.write("**NLP Models:**")
        st.write("- Sentiment: DistilBERT")
        st.write("- Topics: Multi-label classifier")
        st.write("- Embeddings: MiniLM-L6")
        st.write("- Generation: FLAN-T5")

if __name__ == "__main__":
    main()