import re
import string
import unicodedata
from typing import List, Optional
import streamlit as st

class TextPreprocessor:
    """
    Comprehensive text preprocessing for customer feedback analysis
    """
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+\d{1,3}\s?)?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,9}')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        
        # Common contractions
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "mustn't": "must not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will",
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "we'd": "we would",
            "they'd": "they would"
        }
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection (mainly English vs non-English)
        For production, use langdetect or similar libraries
        """
        # Simple heuristic based on common English words
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        words = text.lower().split()
        english_count = sum(1 for word in words if word in english_indicators)
        
        if len(words) == 0:
            return 'unknown'
        
        english_ratio = english_count / len(words)
        return 'english' if english_ratio > 0.1 else 'other'
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        return self.url_pattern.sub(' [URL] ', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses"""
        return self.email_pattern.sub(' [EMAIL] ', text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers"""
        return self.phone_pattern.sub(' [PHONE] ', text)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis while preserving text emoticons like :) :( """
        # First preserve common text emoticons
        text_emoticons = [':)', ':(', ':D', ':P', ';)', ':-)', ':-(', ':-D', ':-P', ';-)']
        placeholders = {}
        
        for i, emoticon in enumerate(text_emoticons):
            if emoticon in text:
                placeholder = f"__EMOTICON_{i}__"
                placeholders[placeholder] = emoticon
                text = text.replace(emoticon, placeholder)
        
        # Remove unicode emojis
        text = self.emoji_pattern.sub('', text)
        
        # Restore text emoticons
        for placeholder, emoticon in placeholders.items():
            text = text.replace(placeholder, emoticon)
        
        return text
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions"""
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Check for contractions (case insensitive)
            lower_word = word.lower()
            if lower_word in self.contractions:
                expanded_words.append(self.contractions[lower_word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and remove extra spaces"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def remove_special_chars(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters while optionally keeping punctuation"""
        if keep_punctuation:
            # Keep basic punctuation: . ! ? , ; : - ( )
            pattern = r'[^\w\s\.\!\?\,\;\:\-\(\)]'
        else:
            # Remove all non-alphanumeric characters except spaces
            pattern = r'[^\w\s]'
        
        return re.sub(pattern, ' ', text)
    
    def fix_common_typos(self, text: str) -> str:
        """Fix common typos and spelling issues"""
        # Common typos in feedback
        typo_fixes = {
            'recieve': 'receive',
            'seperate': 'separate',
            'occured': 'occurred',
            'definately': 'definitely',
            'accomodate': 'accommodate',
            'recomend': 'recommend',
            'experiance': 'experience',
            'custumer': 'customer',
            'sevice': 'service',
            'responce': 'response',
            'usefull': 'useful',
            'helpfull': 'helpful',
            'sucessful': 'successful',
            'proffesional': 'professional',
            'convienient': 'convenient'
        }
        
        words = text.split()
        fixed_words = []
        
        for word in words:
            # Check for typos (case insensitive)
            lower_word = word.lower().strip(string.punctuation)
            if lower_word in typo_fixes:
                # Preserve original case pattern
                if word.isupper():
                    fixed_words.append(typo_fixes[lower_word].upper())
                elif word.istitle():
                    fixed_words.append(typo_fixes[lower_word].title())
                else:
                    fixed_words.append(typo_fixes[lower_word])
            else:
                fixed_words.append(word)
        
        return ' '.join(fixed_words)
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Normalize unicode to decomposed form and then remove combining characters
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        return text
    
    def clean_text(self, text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_phones: bool = True,
                   remove_emojis: bool = True,
                   expand_contractions: bool = True,
                   fix_typos: bool = True,
                   normalize_unicode: bool = True,
                   remove_special_chars: bool = True,
                   keep_punctuation: bool = True,
                   lowercase: bool = True) -> str:
        """
        Comprehensive text cleaning pipeline
        
        Args:
            text: Input text to clean
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_phones: Remove phone numbers
            remove_emojis: Remove emojis
            expand_contractions: Expand contractions
            fix_typos: Fix common typos
            normalize_unicode: Normalize unicode characters
            remove_special_chars: Remove special characters
            keep_punctuation: Keep basic punctuation when removing special chars
            lowercase: Convert to lowercase
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Start with the original text
        cleaned = text
        
        # Apply preprocessing steps
        if normalize_unicode:
            cleaned = self.normalize_unicode(cleaned)
        
        if remove_urls:
            cleaned = self.remove_urls(cleaned)
        
        if remove_emails:
            cleaned = self.remove_emails(cleaned)
        
        if remove_phones:
            cleaned = self.remove_phone_numbers(cleaned)
        
        if remove_emojis:
            cleaned = self.remove_emojis(cleaned)
        
        if expand_contractions:
            cleaned = self.expand_contractions(cleaned)
        
        if fix_typos:
            cleaned = self.fix_common_typos(cleaned)
        
        if remove_special_chars:
            cleaned = self.remove_special_chars(cleaned, keep_punctuation)
        
        # Normalize whitespace
        cleaned = self.normalize_whitespace(cleaned)
        
        if lowercase:
            cleaned = cleaned.lower()
        
        return cleaned
    
    def batch_clean(self, texts: List[str], **kwargs) -> List[str]:
        """
        Clean a batch of texts with the same parameters
        
        Args:
            texts: List of texts to clean
            **kwargs: Arguments to pass to clean_text
        
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text, **kwargs) for text in texts]
    
    def preprocess_feedback_data(self, df, text_column: str = 'feedback', 
                               output_column: str = 'cleaned_text') -> 'pandas.DataFrame':
        """
        Preprocess feedback data in a DataFrame
        
        Args:
            df: pandas DataFrame with feedback data
            text_column: Column name containing the text to clean
            output_column: Column name for the cleaned text
        
        Returns:
            DataFrame with added cleaned text column
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Clean the text
        df[output_column] = df[text_column].apply(self.clean_text)
        
        # Add language detection
        df['detected_language'] = df[text_column].apply(self.detect_language)
        
        # Add text statistics
        df['original_length'] = df[text_column].str.len()
        df['cleaned_length'] = df[output_column].str.len()
        df['word_count'] = df[output_column].str.split().str.len()
        
        return df
    
    def get_preprocessing_stats(self, original_texts: List[str], 
                              cleaned_texts: List[str]) -> dict:
        """
        Get statistics about the preprocessing results
        
        Args:
            original_texts: List of original texts
            cleaned_texts: List of cleaned texts
        
        Returns:
            Dictionary with preprocessing statistics
        """
        if len(original_texts) != len(cleaned_texts):
            raise ValueError("Original and cleaned text lists must have the same length")
        
        stats = {
            'total_texts': len(original_texts),
            'avg_original_length': sum(len(text) for text in original_texts) / len(original_texts),
            'avg_cleaned_length': sum(len(text) for text in cleaned_texts) / len(cleaned_texts),
            'avg_reduction_ratio': 0,
            'empty_after_cleaning': sum(1 for text in cleaned_texts if not text.strip()),
            'language_distribution': {}
        }
        
        # Calculate reduction ratio
        if stats['avg_original_length'] > 0:
            stats['avg_reduction_ratio'] = (
                (stats['avg_original_length'] - stats['avg_cleaned_length']) / 
                stats['avg_original_length']
            )
        
        # Language distribution
        languages = [self.detect_language(text) for text in original_texts]
        for lang in set(languages):
            stats['language_distribution'][lang] = languages.count(lang)
        
        return stats