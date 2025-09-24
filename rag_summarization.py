import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import re
from collections import Counter
import streamlit as st
from embeddings import EmbeddingManager

class RAGSummarizer:
    """
    RAG-based summarization and insight generation
    (Using template-based generation instead of actual LLM for demo)
    """
    
    def __init__(self):
        self.insight_templates = {
            'positive': [
                "Customers appreciate {key_aspects} with feedback highlighting {positive_themes}.",
                "Strong positive sentiment around {key_aspects}, particularly {positive_themes}.",
                "Users are satisfied with {key_aspects}, especially praising {positive_themes}."
            ],
            'negative': [
                "Main concerns center around {key_aspects} with issues related to {negative_themes}.",
                "Customers express dissatisfaction with {key_aspects}, citing problems with {negative_themes}.",
                "Critical feedback focuses on {key_aspects}, particularly {negative_themes}."
            ],
            'mixed': [
                "Feedback shows mixed sentiment on {key_aspects} with both praise and concerns about {themes}.",
                "Customers have varied opinions on {key_aspects} with positive and negative feedback on {themes}."
            ]
        }
        
        self.recommendation_templates = {
            'high_priority': [
                "Immediate action required: Address {issue} to prevent customer churn.",
                "Critical issue: {issue} needs urgent resolution within 1-2 weeks.",
                "High-impact improvement: Focus on {issue} for maximum customer satisfaction gain."
            ],
            'medium_priority': [
                "Important enhancement: Improve {issue} within 2-4 weeks.",
                "Significant opportunity: Address {issue} to boost customer experience.",
                "Notable concern: {issue} requires attention in the next sprint cycle."
            ],
            'low_priority': [
                "Future consideration: {issue} can be addressed in upcoming releases.",
                "Enhancement opportunity: {issue} would improve user experience when resources allow.",
                "Long-term goal: Consider improving {issue} in future product iterations."
            ]
        }
    
    def generate_insights(self, processed_results: List[Dict]) -> Dict:
        """
        Generate AI-powered insights from processed feedback
        
        Args:
            processed_results: List of processed feedback results
        
        Returns:
            Dictionary containing various insights
        """
        insights = {
            'summary': self._generate_executive_summary(processed_results),
            'sentiment_analysis': self._analyze_sentiment_trends(processed_results),
            'topic_insights': self._generate_topic_insights(processed_results),
            'recommendations': self._generate_recommendations(processed_results),
            'emerging_themes': self._identify_emerging_themes(processed_results)
        }
        
        return insights
    
    def _generate_executive_summary(self, processed_results: List[Dict]) -> str:
        """Generate executive summary of feedback"""
        total_feedback = len(processed_results)
        
        # Calculate sentiment distribution
        sentiment_counts = Counter([r['sentiment']['label'] for r in processed_results])
        positive_pct = (sentiment_counts.get('positive', 0) / total_feedback) * 100
        negative_pct = (sentiment_counts.get('negative', 0) / total_feedback) * 100
        
        # Get top topics
        all_topics = []
        for r in processed_results:
            all_topics.extend(r['topics'])
        top_topics = [topic for topic, count in Counter(all_topics).most_common(3)]
        
        # Generate summary
        if positive_pct > 60:
            sentiment_desc = "predominantly positive"
        elif negative_pct > 40:
            sentiment_desc = "concerning with significant negative feedback"
        else:
            sentiment_desc = "mixed but generally balanced"
        
        summary = (
            f"Analysis of {total_feedback} customer feedback entries reveals {sentiment_desc} "
            f"sentiment ({positive_pct:.1f}% positive, {negative_pct:.1f}% negative). "
            f"Primary discussion topics include {', '.join(top_topics[:2])}"
            f"{' and ' + top_topics[2] if len(top_topics) > 2 else ''}. "
        )
        
        if negative_pct > 30:
            summary += "Immediate attention required for critical issues to prevent customer satisfaction decline."
        elif positive_pct > 70:
            summary += "Strong customer satisfaction indicates effective product/service delivery."
        
        return summary
    
    def _analyze_sentiment_trends(self, processed_results: List[Dict]) -> Dict:
        """Analyze sentiment trends and patterns"""
        # Group by time periods (simplified for demo)
        recent_feedback = processed_results[-len(processed_results)//2:] if len(processed_results) > 4 else processed_results
        older_feedback = processed_results[:len(processed_results)//2] if len(processed_results) > 4 else []
        
        if older_feedback:
            recent_positive = sum(1 for r in recent_feedback if r['sentiment']['label'] == 'positive')
            older_positive = sum(1 for r in older_feedback if r['sentiment']['label'] == 'positive')
            
            recent_positive_pct = (recent_positive / len(recent_feedback)) * 100
            older_positive_pct = (older_positive / len(older_feedback)) * 100 if older_feedback else 0
            
            trend = "improving" if recent_positive_pct > older_positive_pct else "declining"
        else:
            recent_positive_pct = sum(1 for r in recent_feedback if r['sentiment']['label'] == 'positive') / len(recent_feedback) * 100
            trend = "stable"
        
        return {
            'overall_trend': trend,
            'recent_positive_rate': round(recent_positive_pct, 1),
            'sentiment_volatility': self._calculate_sentiment_volatility(processed_results)
        }
    
    def _calculate_sentiment_volatility(self, processed_results: List[Dict]) -> float:
        """Calculate sentiment volatility (simplified metric)"""
        compound_scores = [r['sentiment']['compound'] for r in processed_results]
        return round(np.std(compound_scores), 3) if compound_scores else 0.0
    
    def _generate_topic_insights(self, processed_results: List[Dict]) -> List[Dict]:
        """Generate insights for each major topic"""
        # Get all topics with their sentiments
        topic_sentiments = {}
        for result in processed_results:
            for topic in result['topics']:
                if topic not in topic_sentiments:
                    topic_sentiments[topic] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
                
                sentiment = result['sentiment']['label']
                topic_sentiments[topic][sentiment] += 1
                topic_sentiments[topic]['total'] += 1
        
        # Generate insights for top topics
        insights = []
        for topic, counts in sorted(topic_sentiments.items(), key=lambda x: x[1]['total'], reverse=True)[:5]:
            if counts['total'] < 2:  # Skip topics with very few mentions
                continue
                
            positive_pct = (counts['positive'] / counts['total']) * 100
            negative_pct = (counts['negative'] / counts['total']) * 100
            
            if positive_pct > 60:
                insight_type = 'strength'
                description = f"{topic.title()} is a major strength with {positive_pct:.1f}% positive feedback"
            elif negative_pct > 40:
                insight_type = 'concern'
                description = f"{topic.title()} needs attention with {negative_pct:.1f}% negative feedback"
            else:
                insight_type = 'neutral'
                description = f"{topic.title()} shows mixed feedback requiring investigation"
            
            insights.append({
                'topic': topic,
                'type': insight_type,
                'description': description,
                'positive_rate': round(positive_pct, 1),
                'negative_rate': round(negative_pct, 1),
                'total_mentions': counts['total']
            })
        
        return insights
    
    def _generate_recommendations(self, processed_results: List[Dict]) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze by topic and sentiment
        topic_issues = {}
        for result in processed_results:
            if result['sentiment']['label'] == 'negative':
                for topic in result['topics']:
                    if topic not in topic_issues:
                        topic_issues[topic] = []
                    topic_issues[topic].append(result['original_text'])
        
        # Generate recommendations for top issues
        for topic, issues in sorted(topic_issues.items(), key=lambda x: len(x[1]), reverse=True)[:3]:
            issue_count = len(issues)
            
            if issue_count >= len(processed_results) * 0.3:  # 30% or more
                priority = 'high'
                timeline = '1-2 weeks'
                impact = 'Critical for customer retention'
            elif issue_count >= len(processed_results) * 0.15:  # 15% or more
                priority = 'medium'
                timeline = '2-4 weeks'
                impact = 'Significant improvement opportunity'
            else:
                priority = 'low'
                timeline = '1-2 months'
                impact = 'Long-term enhancement'
            
            recommendation = {
                'topic': topic,
                'priority': priority,
                'title': f"Address {topic.replace('_', ' ').title()} Issues",
                'description': self._generate_recommendation_description(topic, issues),
                'expected_impact': impact,
                'timeline': timeline,
                'affected_customers': issue_count
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_recommendation_description(self, topic: str, issues: List[str]) -> str:
        """Generate specific recommendation description"""
        common_keywords = self._extract_common_keywords(issues)
        
        descriptions = {
            'app_performance': f"Improve application stability and speed. Common issues: {', '.join(common_keywords[:3])}.",
            'customer_service': f"Enhance customer support response times and quality. Focus on: {', '.join(common_keywords[:3])}.",
            'user_interface': f"Redesign confusing UI elements and improve user experience. Address: {', '.join(common_keywords[:3])}.",
            'pricing': f"Review pricing strategy and value proposition. Customer concerns: {', '.join(common_keywords[:3])}.",
            'features': f"Enhance or fix problematic features. Priority areas: {', '.join(common_keywords[:3])}.",
            'usability': f"Simplify complex workflows and improve user guidance. Focus on: {', '.join(common_keywords[:3])}."
        }
        
        return descriptions.get(topic, f"Address issues related to {topic}. Common concerns: {', '.join(common_keywords[:3])}.")
    
    def _extract_common_keywords(self, texts: List[str], top_k: int = 5) -> List[str]:
        """Extract common keywords from a list of texts"""
        # Simple keyword extraction
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_words.extend(words)
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'but', 'for', 'are', 'with', 'this', 'that', 'have', 'not', 'was', 'very'}
        filtered_words = [word for word in all_words if word not in stop_words]
        
        # Get most common words
        common_words = [word for word, count in Counter(filtered_words).most_common(top_k)]
        return common_words
    
    def _identify_emerging_themes(self, processed_results: List[Dict]) -> List[Dict]:
        """Identify emerging themes in recent feedback"""
        # For demo, simulate emerging theme detection
        recent_results = processed_results[-len(processed_results)//3:] if len(processed_results) > 6 else processed_results
        
        # Count topics in recent feedback
        recent_topics = []
        for result in recent_results:
            recent_topics.extend(result['topics'])
        
        topic_counts = Counter(recent_topics)
        
        emerging_themes = []
        for topic, count in topic_counts.most_common(3):
            if count >= 2:  # At least 2 mentions to be considered emerging
                theme = {
                    'theme': topic.replace('_', ' ').title(),
                    'mentions': count,
                    'trend': 'increasing',
                    'description': f"Growing customer focus on {topic.replace('_', ' ')} with {count} recent mentions"
                }
                emerging_themes.append(theme)
        
        return emerging_themes
    
    def generate_summary(self, processed_results: List[Dict], 
                        sentiment_filter: str = 'all') -> str:
        """
        Generate a summary for specific sentiment type
        
        Args:
            processed_results: List of processed feedback results
            sentiment_filter: Filter by sentiment ('positive', 'negative', 'neutral', 'all')
        
        Returns:
            Generated summary text
        """
        # Filter results by sentiment if specified
        if sentiment_filter != 'all':
            filtered_results = [r for r in processed_results if r['sentiment']['label'] == sentiment_filter]
        else:
            filtered_results = processed_results
        
        if not filtered_results:
            return f"No {sentiment_filter} feedback found to summarize."
        
        # Extract key themes
        all_topics = []
        for result in filtered_results:
            all_topics.extend(result['topics'])
        
        top_topics = [topic for topic, count in Counter(all_topics).most_common(3)]
        
        # Generate summary based on sentiment
        if sentiment_filter == 'positive':
            summary = (
                f"Positive feedback highlights customer satisfaction with {len(filtered_results)} responses. "
                f"Customers particularly appreciate {', '.join(top_topics[:2])}. "
                f"Common positive themes include good user experience, helpful features, and reliable performance. "
                f"This positive sentiment indicates strong product-market fit and customer loyalty."
            )
        elif sentiment_filter == 'negative':
            summary = (
                f"Negative feedback from {len(filtered_results)} responses reveals key improvement areas. "
                f"Main concerns focus on {', '.join(top_topics[:2])}. "
                f"Critical issues include performance problems, usability challenges, and service gaps. "
                f"Addressing these concerns is essential for customer retention and satisfaction improvement."
            )
        else:
            summary = (
                f"Analysis of {len(filtered_results)} feedback entries shows mixed sentiment. "
                f"Key discussion areas include {', '.join(top_topics)}. "
                f"Customers have varied experiences with both positive highlights and areas needing improvement. "
                f"Balanced approach needed to maintain strengths while addressing concerns."
            )
        
        return summary
    
    def generate_topic_analysis(self, topic_feedback: List[Dict], topic: str) -> str:
        """
        Generate detailed analysis for a specific topic
        
        Args:
            topic_feedback: Feedback results filtered by topic
            topic: Topic name
        
        Returns:
            Topic analysis text
        """
        if not topic_feedback:
            return f"No feedback found for topic: {topic}"
        
        # Sentiment breakdown
        sentiment_counts = Counter([r['sentiment']['label'] for r in topic_feedback])
        total = len(topic_feedback)
        
        positive_pct = (sentiment_counts.get('positive', 0) / total) * 100
        negative_pct = (sentiment_counts.get('negative', 0) / total) * 100
        
        # Extract common phrases
        all_text = ' '.join([r['original_text'] for r in topic_feedback])
        common_words = self._extract_common_keywords([all_text], top_k=5)
        
        # Generate analysis
        analysis = (
            f"Analysis of {total} feedback entries about {topic.replace('_', ' ')} shows "
            f"{positive_pct:.1f}% positive and {negative_pct:.1f}% negative sentiment. "
        )
        
        if positive_pct > 60:
            analysis += f"This topic is generally well-received by customers. "
        elif negative_pct > 40:
            analysis += f"This topic requires immediate attention due to customer concerns. "
        else:
            analysis += f"This topic shows mixed feedback requiring careful analysis. "
        
        analysis += f"Key terms associated with this topic: {', '.join(common_words[:3])}. "
        
        # Add specific recommendations
        if negative_pct > 30:
            analysis += f"Recommend prioritizing improvements in {topic.replace('_', ' ')} to address customer concerns."
        else:
            analysis += f"Consider leveraging positive sentiment around {topic.replace('_', ' ')} as a competitive advantage."
        
        return analysis