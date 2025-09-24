import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

class EmbeddingManager:
    """
    Manages text embeddings and similarity search using TF-IDF
    (In production, this would use sentence-transformers)
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the embedding manager
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        self.embeddings_matrix = None
        self.texts = []
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the vectorizer and transform texts to embeddings
        
        Args:
            texts: List of texts to fit on
        
        Returns:
            Embeddings matrix
        """
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            raise ValueError("No valid texts provided")
        
        self.texts = valid_texts
        self.embeddings_matrix = self.vectorizer.fit_transform(valid_texts).toarray()
        self.is_fitted = True
        
        return self.embeddings_matrix
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts to embeddings using fitted vectorizer
        
        Args:
            texts: List of texts to transform
        
        Returns:
            Embeddings matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        return self.vectorizer.transform(texts).toarray()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        """
        if not self.is_fitted:
            # If not fitted, create a simple mock embedding
            return np.random.rand(100)  # Mock embedding
        
        embedding = self.vectorizer.transform([text]).toarray()
        return embedding[0] if len(embedding) > 0 else np.zeros(self.vectorizer.max_features)
    
    def find_similar(self, query_text: str, top_k: int = 5, 
                     min_similarity: float = 0.1) -> List[Dict]:
        """
        Find similar texts to the query
        
        Args:
            query_text: Query text
            top_k: Number of similar texts to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of dictionaries with similar texts and similarity scores
        """
        if not self.is_fitted:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query_text).reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        
        # Get top-k similar items
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in similar_indices:
            similarity = similarities[idx]
            if similarity >= min_similarity:
                results.append({
                    'text': self.texts[idx],
                    'similarity': round(float(similarity), 4),
                    'index': int(idx)
                })
        
        return results
    
    def cluster_texts(self, n_clusters: int = 5, random_state: int = 42) -> Dict:
        """
        Cluster texts using K-means
        
        Args:
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
        
        Returns:
            Dictionary with cluster results
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings_matrix)
        
        # Organize results
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'text': self.texts[i],
                'index': i
            })
        
        # Get cluster centers and representative texts
        cluster_centers = kmeans.cluster_centers_
        cluster_info = {}
        
        for cluster_id in clusters:
            # Find text closest to cluster center
            cluster_texts_indices = [item['index'] for item in clusters[cluster_id]]
            cluster_embeddings = self.embeddings_matrix[cluster_texts_indices]
            
            center_similarities = cosine_similarity(
                cluster_centers[cluster_id].reshape(1, -1),
                cluster_embeddings
            )[0]
            
            closest_idx = np.argmax(center_similarities)
            representative_text = clusters[cluster_id][closest_idx]['text']
            
            cluster_info[cluster_id] = {
                'size': len(clusters[cluster_id]),
                'texts': clusters[cluster_id],
                'representative_text': representative_text,
                'keywords': self._extract_cluster_keywords(cluster_id, cluster_texts_indices)
            }
        
        return {
            'clusters': cluster_info,
            'labels': cluster_labels.tolist(),
            'n_clusters': n_clusters
        }
    
    def _extract_cluster_keywords(self, cluster_id: int, text_indices: List[int], 
                                top_k: int = 10) -> List[str]:
        """
        Extract top keywords for a cluster
        
        Args:
            cluster_id: Cluster ID
            text_indices: Indices of texts in the cluster
            top_k: Number of top keywords to extract
        
        Returns:
            List of top keywords
        """
        if not self.is_fitted:
            return []
        
        # Get embeddings for cluster texts
        cluster_embeddings = self.embeddings_matrix[text_indices]
        
        # Calculate average TF-IDF scores for the cluster
        avg_scores = np.mean(cluster_embeddings, axis=0)
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top keywords
        top_indices = np.argsort(avg_scores)[::-1][:top_k]
        top_keywords = [feature_names[idx] for idx in top_indices if avg_scores[idx] > 0]
        
        return top_keywords
    
    def get_dimensionality_reduction(self, n_components: int = 2, 
                                   method: str = 'pca') -> np.ndarray:
        """
        Get dimensionally reduced embeddings for visualization
        
        Args:
            n_components: Number of components to reduce to
            method: Dimensionality reduction method ('pca' supported)
        
        Returns:
            Reduced embeddings
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = reducer.fit_transform(self.embeddings_matrix)
            return reduced_embeddings
        else:
            raise ValueError(f"Method '{method}' not supported")
    
    def analyze_semantic_themes(self, min_cluster_size: int = 2) -> Dict:
        """
        Analyze semantic themes in the corpus
        
        Args:
            min_cluster_size: Minimum size for a cluster to be considered a theme
        
        Returns:
            Dictionary with theme analysis
        """
        if not self.is_fitted:
            return {}
        
        # Try different cluster numbers to find optimal themes
        best_clusters = None
        best_score = -1
        
        for n_clusters in range(3, min(10, len(self.texts) // 2)):
            try:
                cluster_result = self.cluster_texts(n_clusters)
                
                # Calculate silhouette-like score (simplified)
                valid_clusters = [
                    cluster for cluster in cluster_result['clusters'].values()
                    if cluster['size'] >= min_cluster_size
                ]
                
                if len(valid_clusters) > best_score:
                    best_score = len(valid_clusters)
                    best_clusters = cluster_result
                    
            except Exception:
                continue
        
        if best_clusters is None:
            return {'themes': [], 'message': 'No clear themes found'}
        
        # Process themes
        themes = []
        for cluster_id, cluster_info in best_clusters['clusters'].items():
            if cluster_info['size'] >= min_cluster_size:
                theme = {
                    'theme_id': cluster_id,
                    'title': f"Theme {cluster_id + 1}",
                    'description': self._generate_theme_description(cluster_info),
                    'keywords': cluster_info['keywords'][:5],
                    'size': cluster_info['size'],
                    'representative_text': cluster_info['representative_text']
                }
                themes.append(theme)
        
        # Sort themes by size
        themes.sort(key=lambda x: x['size'], reverse=True)
        
        return {
            'themes': themes,
            'total_themes': len(themes),
            'coverage': sum(theme['size'] for theme in themes) / len(self.texts)
        }
    
    def _generate_theme_description(self, cluster_info: Dict) -> str:
        """
        Generate a description for a theme based on its keywords
        
        Args:
            cluster_info: Cluster information dictionary
        
        Returns:
            Theme description string
        """
        keywords = cluster_info.get('keywords', [])
        size = cluster_info.get('size', 0)
        
        if not keywords:
            return f"General feedback theme with {size} items"
        
        # Create description based on top keywords
        top_keywords = keywords[:3]
        if len(top_keywords) == 1:
            return f"Feedback primarily about {top_keywords[0]} ({size} items)"
        elif len(top_keywords) == 2:
            return f"Feedback about {top_keywords[0]} and {top_keywords[1]} ({size} items)"
        else:
            return f"Feedback about {', '.join(top_keywords[:-1])}, and {top_keywords[-1]} ({size} items)"


class FAISSManager:
    """
    FAISS-like functionality using scikit-learn for similarity search
    (In production, this would use actual FAISS)
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize FAISS manager
        
        Args:
            embedding_manager: EmbeddingManager instance
        """
        self.embedding_manager = embedding_manager
        self.index_built = False
    
    def build_index(self, embeddings: np.ndarray, texts: List[str]):
        """
        Build search index
        
        Args:
            embeddings: Embedding matrix
            texts: Corresponding texts
        """
        self.embeddings = embeddings
        self.texts = texts
        self.index_built = True
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[int]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
        
        Returns:
            Tuple of (similarities, indices)
        """
        if not self.index_built:
            raise ValueError("Index not built. Call build_index first.")
        
        # Calculate cosine similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[top_indices]
        
        return top_similarities.tolist(), top_indices.tolist()
    
    def add_embeddings(self, new_embeddings: np.ndarray, new_texts: List[str]):
        """
        Add new embeddings to the index
        
        Args:
            new_embeddings: New embedding matrix
            new_texts: Corresponding texts
        """
        if not self.index_built:
            self.build_index(new_embeddings, new_texts)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.texts.extend(new_texts)


class SemanticSearchEngine:
    """
    Semantic search engine combining embedding and keyword search
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize semantic search engine
        
        Args:
            embedding_manager: EmbeddingManager instance
        """
        self.embedding_manager = embedding_manager
        self.faiss_manager = FAISSManager(embedding_manager)
    
    def index_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """
        Index documents for search
        
        Args:
            texts: List of texts to index
            metadata: Optional metadata for each document
        """
        # Get embeddings
        embeddings = self.embedding_manager.fit_transform(texts)
        
        # Build FAISS index
        self.faiss_manager.build_index(embeddings, texts)
        
        # Store metadata
        self.metadata = metadata or [{} for _ in texts]
    
    def search(self, query: str, k: int = 10, 
               include_metadata: bool = True) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            include_metadata: Whether to include metadata in results
        
        Returns:
            List of search results
        """
        # Get query embedding
        query_embedding = self.embedding_manager.get_embedding(query)
        
        # Search using FAISS
        similarities, indices = self.faiss_manager.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities, indices)):
            result = {
                'rank': i + 1,
                'text': self.faiss_manager.texts[idx],
                'similarity': round(similarity, 4),
                'index': idx
            }
            
            if include_metadata and hasattr(self, 'metadata'):
                result['metadata'] = self.metadata[idx]
            
            results.append(result)
        
        return results
    
    def hybrid_search(self, query: str, k: int = 10, 
                     semantic_weight: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic similarity (0-1)
        
        Returns:
            List of hybrid search results
        """
        # Get semantic results
        semantic_results = self.search(query, k * 2)  # Get more for reranking
        
        # Simple keyword matching
        query_words = set(query.lower().split())
        keyword_scores = []
        
        for result in semantic_results:
            text_words = set(result['text'].lower().split())
            keyword_overlap = len(query_words.intersection(text_words))
            keyword_score = keyword_overlap / len(query_words) if query_words else 0
            keyword_scores.append(keyword_score)
        
        # Combine scores
        combined_results = []
        for i, result in enumerate(semantic_results):
            combined_score = (
                semantic_weight * result['similarity'] +
                (1 - semantic_weight) * keyword_scores[i]
            )
            
            result['combined_score'] = round(combined_score, 4)
            result['keyword_score'] = round(keyword_scores[i], 4)
            combined_results.append(result)
        
        # Sort by combined score and return top-k
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results[:k]):
            result['rank'] = i + 1
        
        return combined_results[:k]