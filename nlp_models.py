import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

class FeedbackAnalyzer:
    """
    Comprehensive feedback analysis using various NLP techniques
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_classifier = TopicClassifier()
        self.aspect_analyzer = AspectBasedSentimentAnalyzer()
        self.emotion_detector = EmotionDetector()
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        return self.sentiment_analyzer.analyze(text)
    
    def classify_topics(self, text: str) -> List[str]:
        """Classify topics in text"""
        return self.topic_classifier.classify(text)
    
    def analyze_aspects(self, text: str) -> List[Dict]:
        """Perform aspect-based sentiment analysis"""
        return self.aspect_analyzer.analyze(text)
    
    def detect_emotions(self, text: str) -> Dict:
        """Detect emotions in text"""
        return self.emotion_detector.detect(text)
    
    def get_comprehensive_analysis(self, text: str) -> Dict:
        """Get comprehensive analysis of text"""
        return {
            'sentiment': self.analyze_sentiment(text),
            'topics': self.classify_topics(text),
            'aspects': self.analyze_aspects(text),
            'emotions': self.detect_emotions(text)
        }


class SentimentAnalyzer:
    """
    Sentiment analysis using lexicon-based approach with ML-like scoring
    """
    
    def __init__(self):
        # Positive words
        self.positive_words = {
            'excellent', 'amazing', 'fantastic', 'great', 'good', 'wonderful', 
            'awesome', 'brilliant', 'outstanding', 'superb', 'perfect', 'love',
            'like', 'enjoy', 'satisfied', 'happy', 'pleased', 'impressed',
            'recommend', 'helpful', 'useful', 'easy', 'convenient', 'fast',
            'quick', 'efficient', 'smooth', 'reliable', 'stable', 'secure'
        }
        
        # Negative words
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'dislike',
            'annoying', 'frustrating', 'disappointed', 'unsatisfied', 'angry',
            'upset', 'confused', 'difficult', 'hard', 'complicated', 'slow',
            'buggy', 'broken', 'crash', 'error', 'problem', 'issue', 'fail',
            'useless', 'worthless', 'expensive', 'overpriced', 'poor', 'lacking'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'really': 1.4, 'extremely': 1.8, 'incredibly': 1.6,
            'absolutely': 1.7, 'completely': 1.5, 'totally': 1.4, 'quite': 1.2,
            'rather': 1.1, 'somewhat': 0.8, 'slightly': 0.7, 'barely': 0.5
        }
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither',
            'nowhere', 'cannot', 'cant', 'wont', 'shouldnt', 'wouldnt',
            'couldnt', 'doesnt', 'dont', 'isnt', 'arent', 'wasnt', 'werent'
        }
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not text.strip():
            return {
                'label': 'neutral',
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'confidence': 0.0
            }
        
        words = text.lower().split()
        total_score = 0.0
        word_count = 0
        
        i = 0
        while i < len(words):
            word = words[i].strip('.,!?;:"()[]{}')
            
            # Check for intensifiers
            intensity = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensity = self.intensifiers[words[i-1]]
            
            # Check for negation
            negated = False
            if i > 0 and words[i-1] in self.negation_words:
                negated = True
            elif i > 1 and words[i-2] in self.negation_words:
                negated = True
            
            # Calculate word sentiment
            word_sentiment = 0.0
            if word in self.positive_words:
                word_sentiment = 1.0
            elif word in self.negative_words:
                word_sentiment = -1.0
            
            # Apply intensity and negation
            if word_sentiment != 0:
                word_sentiment *= intensity
                if negated:
                    word_sentiment *= -0.8  # Negation doesn't completely flip
                
                total_score += word_sentiment
                word_count += 1
            
            i += 1
        
        # Normalize score
        if word_count > 0:
            compound = total_score / word_count
        else:
            compound = 0.0
        
        # Clamp compound score
        compound = max(-1.0, min(1.0, compound))
        
        # Calculate positive, negative, neutral probabilities
        if compound >= 0.05:
            label = 'positive'
            positive = (compound + 1) / 2
            negative = max(0, (1 - positive) / 2)
            neutral = 1 - positive - negative
        elif compound <= -0.05:
            label = 'negative'
            negative = (abs(compound) + 1) / 2
            positive = max(0, (1 - negative) / 2)
            neutral = 1 - positive - negative
        else:
            label = 'neutral'
            neutral = 0.6 + 0.4 * (1 - abs(compound) * 10)  # Higher neutral for scores close to 0
            positive = (1 - neutral) / 2
            negative = (1 - neutral) / 2
        
        # Calculate confidence
        confidence = abs(compound) if abs(compound) > 0.05 else 0.1
        confidence = min(0.95, max(0.1, confidence))
        
        return {
            'label': label,
            'compound': round(compound, 3),
            'positive': round(positive, 3),
            'negative': round(negative, 3),
            'neutral': round(neutral, 3),
            'confidence': round(confidence, 3)
        }


class TopicClassifier:
    """
    Topic classification using keyword-based approach
    """
    
    def __init__(self):
        self.topic_keywords = {
            'app_performance': {
                'crash', 'slow', 'lag', 'freeze', 'bug', 'error', 'performance',
                'speed', 'loading', 'response', 'hang', 'timeout', 'glitch'
            },
            'user_interface': {
                'ui', 'interface', 'design', 'layout', 'navigation', 'menu',
                'button', 'screen', 'display', 'visual', 'color', 'theme',
                'font', 'size', 'responsive', 'mobile', 'desktop'
            },
            'customer_service': {
                'support', 'help', 'service', 'staff', 'representative', 'agent',
                'response', 'assistance', 'contact', 'phone', 'email', 'chat',
                'ticket', 'resolution', 'follow-up'
            },
            'pricing': {
                'price', 'cost', 'expensive', 'cheap', 'affordable', 'value',
                'money', 'payment', 'billing', 'subscription', 'plan', 'fee',
                'discount', 'offer', 'deal', 'refund'
            },
            'features': {
                'feature', 'function', 'functionality', 'capability', 'option',
                'setting', 'tool', 'integration', 'api', 'export', 'import',
                'sync', 'backup', 'security', 'privacy'
            },
            'usability': {
                'easy', 'difficult', 'hard', 'simple', 'complex', 'intuitive',
                'confusing', 'user-friendly', 'experience', 'workflow',
                'process', 'step', 'guide', 'tutorial', 'documentation'
            },
            'delivery': {
                'shipping', 'delivery', 'package', 'order', 'tracking',
                'arrival', 'delay', 'fast', 'quick', 'logistics', 'courier'
            },
            'quality': {
                'quality', 'build', 'material', 'durability', 'reliability',
                'craftsmanship', 'finish', 'design', 'construction'
            }
        }
    
    def classify(self, text: str, threshold: float = 0.1) -> List[str]:
        """
        Classify topics in text
        
        Args:
            text: Input text
            threshold: Minimum score threshold for topic classification
        
        Returns:
            List of detected topics
        """
        if not text or not text.strip():
            return []
        
        words = set(text.lower().split())
        topic_scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            # Count keyword matches
            matches = len(words.intersection(keywords))
            # Calculate score as ratio of matches to total words
            score = matches / len(words) if len(words) > 0 else 0
            
            if score >= threshold:
                topic_scores[topic] = score
        
        # Sort topics by score and return
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics]


class AspectBasedSentimentAnalyzer:
    """
    Aspect-based sentiment analysis
    """
    
    def __init__(self):
        self.aspects = {
            'performance': {
                'keywords': ['speed', 'fast', 'slow', 'performance', 'lag', 'quick', 'responsive'],
                'sentiment_words': {
                    'positive': ['fast', 'quick', 'responsive', 'smooth', 'efficient'],
                    'negative': ['slow', 'lag', 'sluggish', 'unresponsive', 'delayed']
                }
            },
            'design': {
                'keywords': ['design', 'ui', 'interface', 'layout', 'visual', 'appearance'],
                'sentiment_words': {
                    'positive': ['beautiful', 'clean', 'modern', 'intuitive', 'attractive'],
                    'negative': ['ugly', 'cluttered', 'outdated', 'confusing', 'messy']
                }
            },
            'functionality': {
                'keywords': ['feature', 'function', 'work', 'working', 'functionality'],
                'sentiment_words': {
                    'positive': ['useful', 'helpful', 'powerful', 'complete', 'robust'],
                    'negative': ['broken', 'useless', 'limited', 'missing', 'incomplete']
                }
            },
            'support': {
                'keywords': ['support', 'help', 'service', 'assistance', 'customer'],
                'sentiment_words': {
                    'positive': ['helpful', 'responsive', 'professional', 'knowledgeable'],
                    'negative': ['unhelpful', 'rude', 'slow', 'incompetent', 'unresponsive']
                }
            },
            'value': {
                'keywords': ['price', 'cost', 'value', 'money', 'worth', 'expensive'],
                'sentiment_words': {
                    'positive': ['affordable', 'reasonable', 'worth', 'value', 'cheap'],
                    'negative': ['expensive', 'overpriced', 'costly', 'waste', 'ripoff']
                }
            }
        }
    
    def analyze(self, text: str) -> List[Dict]:
        """
        Analyze aspects and their sentiments
        
        Returns:
            List of dictionaries with aspect and sentiment information
        """
        if not text or not text.strip():
            return []
        
        words = text.lower().split()
        word_set = set(words)
        results = []
        
        for aspect_name, aspect_data in self.aspects.items():
            # Check if aspect is mentioned
            aspect_mentioned = bool(word_set.intersection(aspect_data['keywords']))
            
            if aspect_mentioned:
                # Calculate sentiment for this aspect
                pos_matches = len(word_set.intersection(aspect_data['sentiment_words']['positive']))
                neg_matches = len(word_set.intersection(aspect_data['sentiment_words']['negative']))
                
                if pos_matches > neg_matches:
                    sentiment = 'positive'
                    confidence = pos_matches / (pos_matches + neg_matches) if (pos_matches + neg_matches) > 0 else 0.5
                elif neg_matches > pos_matches:
                    sentiment = 'negative'
                    confidence = neg_matches / (pos_matches + neg_matches) if (pos_matches + neg_matches) > 0 else 0.5
                else:
                    sentiment = 'neutral'
                    confidence = 0.5
                
                results.append({
                    'aspect': aspect_name,
                    'sentiment': sentiment,
                    'confidence': round(confidence, 3),
                    'mentions': list(word_set.intersection(aspect_data['keywords']))
                })
        
        return results


class EmotionDetector:
    """
    Emotion detection using keyword-based approach
    """
    
    def __init__(self):
        self.emotion_keywords = {
            'joy': {
                'love', 'happy', 'excited', 'thrilled', 'delighted', 'pleased',
                'satisfied', 'glad', 'cheerful', 'joyful', 'ecstatic'
            },
            'anger': {
                'angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated',
                'outraged', 'livid', 'infuriated', 'rage', 'hate'
            },
            'sadness': {
                'sad', 'disappointed', 'upset', 'depressed', 'dejected',
                'heartbroken', 'miserable', 'unhappy', 'sorrowful', 'gloomy'
            },
            'fear': {
                'scared', 'afraid', 'worried', 'anxious', 'nervous', 'concerned',
                'terrified', 'frightened', 'apprehensive', 'uneasy', 'panic'
            },
            'surprise': {
                'surprised', 'amazed', 'astonished', 'shocked', 'stunned',
                'bewildered', 'startled', 'unexpected', 'incredible', 'wow'
            },
            'disgust': {
                'disgusted', 'revolted', 'repulsed', 'appalled', 'horrified',
                'sickened', 'nauseated', 'gross', 'awful', 'terrible'
            }
        }
    
    def detect(self, text: str) -> Dict:
        """
        Detect emotions in text
        
        Returns:
            Dictionary with emotion scores
        """
        if not text or not text.strip():
            return {emotion: 0.0 for emotion in self.emotion_keywords}
        
        words = set(text.lower().split())
        emotion_scores = {}
        total_matches = 0
        
        for emotion, keywords in self.emotion_keywords.items():
            matches = len(words.intersection(keywords))
            emotion_scores[emotion] = matches
            total_matches += matches
        
        # Normalize scores
        if total_matches > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = round(emotion_scores[emotion] / total_matches, 3)
        else:
            emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords}
        
        return emotion_scores