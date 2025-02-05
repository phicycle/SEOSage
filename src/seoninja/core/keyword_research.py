"""Core keyword research functionality."""
import requests
import logging
from typing import List, Dict, Any, Optional
import uuid
import json
import os
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import aiohttp
import asyncio
from pathlib import Path
from functools import lru_cache
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from aiohttp import ClientTimeout

class KeywordResearch:
    """Enhanced keyword research with improved analysis capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self._api_key = self.config.get('api_key')
        self._api_base_url = self.config.get('api_base_url', 'https://api.keywordtool.io/v1')
        self._batch_size = self.config.get('batch_size', 50)
        self._rate_limit = self.config.get('rate_limit', 1.0)  # Requests per second
        
        # Cache configuration
        self._cache_dir = Path('data/cache/keywords')
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_duration = timedelta(days=30)
        
        # Initialize metrics
        self._metrics: Dict[str, float] = defaultdict(float)
        
        # Initialize session
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0
        
    async def _init_session(self) -> None:
        """Initialize API session with retry logic."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'Authorization': f'Bearer {self._api_key}'}
            )
            
    async def analyze_keywords(
        self,
        keywords: List[str],
        competitors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze keywords with batching and caching."""
        await self._init_session()
        
        results = {
            'keywords': {},
            'metrics': {},
            'suggestions': [],
            'competitors': {}
        }
        
        # Process keywords in batches
        for i in range(0, len(keywords), self._batch_size):
            batch = keywords[i:i + self._batch_size]
            batch_results = await self._process_keyword_batch(batch)
            results['keywords'].update(batch_results)
            
        # Analyze competitors if provided
        if competitors:
            results['competitors'] = await self._analyze_competitors(
                keywords,
                competitors
            )
            
        # Generate keyword suggestions
        results['suggestions'] = await self._generate_suggestions(keywords)
        
        # Calculate aggregate metrics
        results['metrics'] = self._calculate_metrics(results)
        
        return results
        
    async def _process_keyword_batch(
        self,
        keywords: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Process a batch of keywords."""
        results = {}
        cache_hits = 0
        
        for keyword in keywords:
            # Check cache first
            cached_result = self._get_from_cache(keyword)
            if cached_result:
                results[keyword] = cached_result
                cache_hits += 1
                continue
                
            # Respect rate limiting
            await self._respect_rate_limit()
            
            try:
                # Get keyword metrics
                metrics = await self._get_keyword_metrics(keyword)
                
                # Get search intent
                intent = await self._analyze_search_intent(keyword)
                
                # Get related keywords
                related = await self._get_related_keywords(keyword)
                
                # Combine results
                result = {
                    'metrics': metrics,
                    'intent': intent,
                    'related': related,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Cache result
                self._save_to_cache(keyword, result)
                
                results[keyword] = result
                
            except Exception as e:
                self.logger.error(f"Error processing keyword {keyword}: {str(e)}")
                results[keyword] = {'error': str(e)}
                
        self._metrics['cache_hit_rate'] = cache_hits / len(keywords)
        
        return results
        
    async def _get_keyword_metrics(self, keyword: str) -> Dict[str, Any]:
        """Get keyword metrics from API."""
        if not self._session:
            raise RuntimeError("Session not initialized")
            
        url = f"{self._api_base_url}/metrics"
        params = {'keyword': keyword}
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API error: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error getting metrics for {keyword}: {str(e)}")
            return {}
            
    async def _analyze_search_intent(self, keyword: str) -> Dict[str, float]:
        """Analyze search intent for keyword."""
        # Intent categories
        intents = {
            'informational': 0.0,
            'navigational': 0.0,
            'transactional': 0.0,
            'commercial': 0.0
        }
        
        # Intent signals
        signals = {
            'informational': ['how', 'what', 'why', 'when', 'where', 'guide', 'tutorial'],
            'navigational': ['login', 'sign in', 'website', 'official'],
            'transactional': ['buy', 'price', 'order', 'purchase', 'cheap', 'deal'],
            'commercial': ['best', 'review', 'top', 'vs', 'compare']
        }
        
        # Calculate intent scores
        words = keyword.lower().split()
        for intent, keywords in signals.items():
            score = sum(1 for word in words if word in keywords)
            intents[intent] = score / len(words) if words else 0.0
            
        # Normalize scores
        total = sum(intents.values())
        if total > 0:
            for intent in intents:
                intents[intent] /= total
                
        return intents
        
    async def _get_related_keywords(self, keyword: str) -> List[Dict[str, Any]]:
        """Get related keywords from API."""
        if not self._session:
            raise RuntimeError("Session not initialized")
            
        url = f"{self._api_base_url}/related"
        params = {'keyword': keyword, 'limit': 10}
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API error: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error getting related keywords for {keyword}: {str(e)}")
            return []
            
    async def _analyze_competitors(
        self,
        keywords: List[str],
        competitors: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze competitor keyword usage."""
        results = {}
        
        for competitor in competitors:
            try:
                # Get competitor content
                content = await self._fetch_competitor_content(competitor)
                
                # Analyze keyword usage
                keyword_usage = self._analyze_keyword_usage(content, keywords)
                
                # Get content topics
                topics = self._extract_topics(content)
                
                results[competitor] = {
                    'keyword_usage': keyword_usage,
                    'topics': topics,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing competitor {competitor}: {str(e)}")
                results[competitor] = {'error': str(e)}
                
        return results
        
    async def _fetch_competitor_content(self, url: str) -> str:
        """Fetch competitor website content."""
        if not self._session:
            raise RuntimeError("Session not initialized")
            
        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    raise Exception(f"Failed to fetch content: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error fetching content from {url}: {str(e)}")
            return ""
            
    def _analyze_keyword_usage(
        self,
        content: str,
        keywords: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze keyword usage in content."""
        results = {}
        
        # Normalize content
        content_lower = content.lower()
        words = content_lower.split()
        total_words = len(words)
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            
            results[keyword] = {
                'count': count,
                'density': count / total_words if total_words > 0 else 0.0,
                'prominence': self._calculate_prominence(content_lower, keyword_lower)
            }
            
        return results
        
    def _calculate_prominence(self, content: str, keyword: str) -> float:
        """Calculate keyword prominence in content."""
        # Prominence factors:
        # - Appearance in title/headers
        # - Position in content
        # - Density in important sections
        
        prominence = 0.0
        
        # Check position of first occurrence
        first_pos = content.find(keyword)
        if first_pos >= 0:
            # Higher score for earlier appearances
            prominence += 1.0 - (first_pos / len(content))
            
        # Add bonus for title/header appearances
        if keyword in content[:100]:  # Approximate title/header area
            prominence += 0.5
            
        return min(prominence, 1.0)
        
    def _extract_topics(self, content: str) -> List[Dict[str, Any]]:
        """Extract main topics from content."""
        # Use TF-IDF to identify important terms
        vectorizer = TfidfVectorizer(
            max_features=20,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            # Transform content
            tfidf_matrix = vectorizer.fit_transform([content])
            
            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Sort by importance
            topics = [
                {
                    'topic': feature_names[i],
                    'score': float(scores[i])
                }
                for i in np.argsort(scores)[-10:][::-1]
            ]
            
            return topics
            
        except Exception as e:
            self.logger.error(f"Error extracting topics: {str(e)}")
            return []
            
    async def _generate_suggestions(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Generate keyword suggestions."""
        suggestions = []
        
        try:
            # Get suggestions from API
            url = f"{self._api_base_url}/suggestions"
            params = {'keywords': ','.join(keywords), 'limit': 20}
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    api_suggestions = await response.json()
                    suggestions.extend(api_suggestions)
                    
            # Add variations of existing keywords
            variations = self._generate_keyword_variations(keywords)
            suggestions.extend(variations)
            
            # Remove duplicates and sort by score
            suggestions = self._deduplicate_suggestions(suggestions)
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {str(e)}")
            
        return suggestions
        
    def _generate_keyword_variations(
        self,
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate variations of keywords."""
        variations = []
        modifiers = ['best', 'top', 'cheap', 'online', 'free', 'professional']
        
        for keyword in keywords:
            # Add modifiers
            for modifier in modifiers:
                variations.append({
                    'keyword': f"{modifier} {keyword}",
                    'source': 'variation',
                    'score': 0.5
                })
                variations.append({
                    'keyword': f"{keyword} {modifier}",
                    'source': 'variation',
                    'score': 0.5
                })
                
        return variations
        
    def _deduplicate_suggestions(
        self,
        suggestions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate suggestions and sort by score."""
        # Use dict to remove duplicates while keeping highest score
        unique_suggestions = {}
        
        for suggestion in suggestions:
            keyword = suggestion['keyword']
            if (
                keyword not in unique_suggestions or
                suggestion['score'] > unique_suggestions[keyword]['score']
            ):
                unique_suggestions[keyword] = suggestion
                
        # Convert back to list and sort by score
        return sorted(
            unique_suggestions.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
    def _calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate metrics from results."""
        metrics = {
            'total_keywords': len(results['keywords']),
            'avg_competition': 0.0,
            'avg_search_volume': 0.0,
            'intent_distribution': defaultdict(float)
        }
        
        # Calculate averages
        for keyword, data in results['keywords'].items():
            if 'metrics' in data:
                metrics['avg_competition'] += data['metrics'].get('competition', 0)
                metrics['avg_search_volume'] += data['metrics'].get('search_volume', 0)
                
            if 'intent' in data:
                for intent, score in data['intent'].items():
                    metrics['intent_distribution'][intent] += score
                    
        # Normalize averages
        if metrics['total_keywords'] > 0:
            metrics['avg_competition'] /= metrics['total_keywords']
            metrics['avg_search_volume'] /= metrics['total_keywords']
            
            for intent in metrics['intent_distribution']:
                metrics['intent_distribution'][intent] /= metrics['total_keywords']
                
        return metrics
        
    async def _respect_rate_limit(self) -> None:
        """Respect API rate limiting."""
        if self._rate_limit > 0:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            if time_since_last < self._rate_limit:
                await asyncio.sleep(self._rate_limit - time_since_last)
                
            self._last_request_time = time.time()
            
    def _get_cache_key(self, keyword: str) -> str:
        """Generate cache key for keyword."""
        return hashlib.md5(keyword.encode()).hexdigest()
        
    def _get_from_cache(self, keyword: str) -> Optional[Dict[str, Any]]:
        """Get cached keyword results."""
        cache_key = self._get_cache_key(keyword)
        cache_file = self._cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            data = json.loads(cache_file.read_text())
            cached_time = datetime.fromisoformat(data['timestamp'])
            
            if datetime.now() - cached_time > self._cache_duration:
                return None
                
            return data
            
        except Exception as e:
            self.logger.error(f"Cache read error: {str(e)}")
            return None
            
    def _save_to_cache(self, keyword: str, data: Dict[str, Any]) -> None:
        """Save keyword results to cache."""
        cache_key = self._get_cache_key(keyword)
        cache_file = self._cache_dir / f"{cache_key}.json"
        
        try:
            cache_file.write_text(json.dumps(data))
            
        except Exception as e:
            self.logger.error(f"Cache write error: {str(e)}")
            
    def clear_cache(self) -> None:
        """Clear keyword cache."""
        try:
            for cache_file in self._cache_dir.glob('*.json'):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return dict(self._metrics)
        
    async def close(self) -> None:
        """Close resources."""
        if self._session:
            await self._session.close()
            self._session = None 