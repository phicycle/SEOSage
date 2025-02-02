import requests
import logging
from typing import List, Dict, Any, Optional
import uuid
import json
import os
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict

class KeywordResearch:
    """
    Performs keyword research and analysis using the Moz API (JSON-RPC 2.0).
    
    Attributes:
        api_token: Moz API token
        base_url: Moz API base URL
        logger: Logger instance
        monthly_quota: Free tier quota
        _quota_used: Track quota usage
    """
    
    def __init__(self, api_token: str) -> None:
        """Initialize keyword research tool with API token."""
        if not api_token:
            raise ValueError("API token cannot be empty")
        
        self.api_token = api_token.strip()
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"KeywordResearch initialized with token length: {len(self.api_token)}")
        
        # Update to use JSON-RPC 2.0 endpoint
        self.base_url = "https://api.moz.com/jsonrpc"
        self.monthly_quota = 50
        self._quota_used = 0
        
        # Initialize cache settings
        self.cache_dir = "cache/moz_api"
        self.cache_duration = timedelta(days=7)
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _make_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a JSON-RPC 2.0 request to the Moz API.
        
        Args:
            method: The RPC method name
            params: Method-specific parameters
            
        Returns:
            API response data
        """
        # Generate cache key
        cache_key = self._get_cache_key(method, params)
        
        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response is not None:
            return cached_response
            
        headers = {
            'x-moz-token': self.api_token,
            'Content-Type': 'application/json'
        }
        
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": {"data": params}
        }
        
        try:
            self.logger.info(f"Making API request to method: {method}")
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            self.logger.debug(f"API Response Status: {response.status_code}")
            self.logger.debug(f"API Response: {response.text}")
            
            response.raise_for_status()
            data = response.json()
            result = data.get('result', {})
            
            # Update quota usage from response
            if 'quota_used' in result:
                self._quota_used += result['quota_used']
            
            # Save to cache
            self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"Response content: {e.response.text}")
            raise

    def get_ranking_keywords(self, domain: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get ranking keywords for a domain using the data.site.ranking.keywords.list method.
        
        Args:
            domain: The domain to analyze
            limit: Maximum number of keywords to return
            
        Returns:
            List of ranking keywords with metrics
        """
        params = {
            "target_query": {
                "query": domain,
                "scope": "domain",
                "locale": "en-US"
            },
            "page": {
                "n": 0,
                "limit": limit
            },
            "options": {
                "sort": "rank",
            }
        }
        
        try:
            data = self._make_request("data.site.ranking.keywords.list", params)
            return self._process_ranking_keywords(data)
        except Exception as e:
            self.logger.error(f"Failed to get ranking keywords: {str(e)}")
            return []

    def _process_ranking_keywords(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process ranking keywords response."""
        keywords = []
        for item in data.get('ranking_keywords', []):
            try:
                keyword = {
                    'keyword': item.get('keyword', ''),
                    'rank_position': item.get('rank_position', 0),
                    'difficulty': item.get('difficulty', 0),
                    'volume': item.get('volume', 0),
                    'ranking_page': item.get('ranking_page', ''),
                    'opportunity_score': self._calculate_opportunity_score(
                        rank=item.get('rank_position', 0),
                        volume=item.get('volume', 0),
                        difficulty=item.get('difficulty', 0)
                    )
                }
                keywords.append(keyword)
            except Exception as e:
                self.logger.warning(f"Error processing keyword: {str(e)}")
                continue
        return keywords

    def get_keyword_metrics(self, keyword: str) -> Dict[str, Any]:
        """
        Get metrics for a specific keyword using data.keyword.metrics.fetch.
        
        Args:
            keyword: The keyword to analyze
            
        Returns:
            Dictionary of keyword metrics
        """
        params = {
            "serp_query": {
                "keyword": keyword,
                "locale": "en-US",
                "device": "desktop",
                "engine": "google"
            }
        }
        
        try:
            data = self._make_request("data.keyword.metrics.fetch", params)
            return self._process_keyword_metrics(data)
        except Exception as e:
            self.logger.error(f"Failed to get keyword metrics: {str(e)}")
            return {}

    def _process_keyword_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process keyword metrics response."""
        metrics = data.get('keyword_metrics', {})
        return {
            'volume': metrics.get('volume', 0) or 0,
            'difficulty': metrics.get('difficulty', 0) or 0,
            'organic_ctr': metrics.get('organic_ctr', 0) or 0,
            'priority': metrics.get('priority', 0) or 0
        }

    def get_keyword_suggestions(self, keyword: str, limit: int = 8) -> List[Dict[str, Any]]:
        """
        Get keyword suggestions using data.keyword.suggestions.list.
        
        Args:
            keyword: Seed keyword
            limit: Maximum number of suggestions to return
            
        Returns:
            List of keyword suggestions with relevance scores
        """
        params = {
            "serp_query": {
                "keyword": keyword,
                "locale": "en-US",
                "device": "desktop",
                "engine": "google"
            },
            "page": {
                "n": 0,
                "limit": limit
            }
        }
        
        try:
            data = self._make_request("data.keyword.suggestions.list", params)
            return self._process_keyword_suggestions(data)
        except Exception as e:
            self.logger.error(f"Failed to get keyword suggestions: {str(e)}")
            return []

    def _process_keyword_suggestions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process keyword suggestions response."""
        suggestions = []
        for item in data.get('suggestions', []):
            try:
                suggestion = {
                    'keyword': item.get('keyword', ''),
                    'relevance': item.get('relevance', 0),
                    'metrics': self.get_keyword_metrics(item.get('keyword', ''))
                }
                suggestions.append(suggestion)
            except Exception as e:
                self.logger.warning(f"Error processing suggestion: {str(e)}")
                continue
        return suggestions

    def get_quota(self) -> Dict[str, Any]:
        """
        Get current quota usage using quota.lookup.
        
        Returns:
            Dictionary with quota information
        """
        params = {
            "path": "api.limits.data.rows"
        }
        
        try:
            return self._make_request("quota.lookup", params)
        except Exception as e:
            self.logger.error(f"Failed to get quota: {str(e)}")
            return {}

    def _calculate_opportunity_score(self, rank: int, volume: int, difficulty: float) -> float:
        """
        Calculate opportunity score based on keyword metrics.
        
        Args:
            rank: Current ranking position
            volume: Monthly search volume
            difficulty: Keyword difficulty (0-100)
            
        Returns:
            Opportunity score (0-100)
        """
        # Normalize values to 0-1 scale
        volume_score = min(volume / 10000, 1.0)
        difficulty_score = 1 - (difficulty / 100)
        rank_score = 1 - (min(rank, 30) / 30)
        
        # Weighted score
        return round(
            (volume_score * 0.4) + 
            (difficulty_score * 0.3) + 
            (rank_score * 0.3), 
            2
        ) * 100
    def get_ranking_opportunities(self, domain: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get ranking opportunities for a domain using the new Moz API.
        
        Args:
            domain: The domain to analyze
            limit: Maximum number of opportunities to return
            
        Returns:
            List of ranking opportunities with metrics
        """
        try:
            # Get ranking keywords
            keywords = self.get_ranking_keywords(domain, limit=limit)
            if not keywords:
                return []
                
            # Filter and analyze opportunities
            opportunities = []
            for kw in keywords:
                try:
                    if (4 <= kw['rank_position'] <= 30 and
                        kw['volume'] >= 100 and
                        kw['difficulty'] <= 60):
                        
                        # Get additional metrics
                        metrics = self.get_keyword_metrics(kw['keyword'])
                        if metrics.get('organic_ctr', 0) >= 0.1:
                            kw.update(metrics)
                            kw['opportunity_score'] = self._calculate_opportunity_score(
                                rank=kw['rank_position'],
                                volume=kw['volume'],
                                difficulty=kw['difficulty']
                            )
                            opportunities.append(kw)
                            
                except KeyError as e:
                    self.logger.warning(f"Missing required metric for keyword analysis: {str(e)}")
                    continue
                
            # Sort and return top opportunities
            return sorted(
                opportunities,
                key=lambda x: x['opportunity_score'],
                reverse=True
            )[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get ranking opportunities: {str(e)}")
            return []

    def get_related_keywords(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get related keywords using data.keyword.suggestions.list.
        
        Args:
            keyword: Seed keyword
            limit: Maximum number of related keywords to return
            
        Returns:
            List of related keywords with metrics
        """
        params = {
            "serp_query": {
                "keyword": keyword,
                "locale": "en-US",
                "device": "desktop",
                "engine": "google"
            },
            "page": {
                "n": 0,
                "limit": limit
            }
        }
        
        try:
            # Update method name to use the correct beta endpoint
            data = self._make_request("data.keyword.suggestions.list", params)
            return self._process_related_keywords(data)
        except Exception as e:
            self.logger.error(f"Failed to get related keywords: {str(e)}")
            return []

    def _process_related_keywords(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process related keywords response."""
        keywords = []
        for item in data.get('suggestions', []):
            try:
                keyword = {
                    'keyword': item.get('keyword', ''),
                    'relevance': item.get('relevance', 0),
                    'metrics': self.get_keyword_metrics(item.get('keyword', ''))
                }
                keywords.append(keyword)
            except Exception as e:
                self.logger.warning(f"Error processing related keyword: {str(e)}")
                continue
        return keywords

    def get_keyword_analysis(self, keyword: str) -> Dict[str, Any]:
        """
        Get comprehensive keyword analysis using multiple API methods.
        
        Args:
            keyword: The keyword to analyze
            
        Returns:
            Dictionary with comprehensive keyword analysis
        """
        try:
            # Get basic metrics
            metrics = self.get_keyword_metrics(keyword)
            
            # Get suggestions and related keywords
            suggestions = self.get_keyword_suggestions(keyword)
            related = self.get_related_keywords(keyword)
            
            # Get SERP analysis
            serp = self.get_serp_analysis(keyword)
            
            return {
                'metrics': metrics,
                'suggestions': suggestions,
                'related_keywords': related,
                'serp_analysis': serp
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get keyword analysis: {str(e)}")
            return {}

    def get_serp_analysis(self, keyword: str) -> Dict[str, Any]:
        """
        Get SERP analysis for a keyword using data.serp.analysis.fetch.
        
        Args:
            keyword: The keyword to analyze
            
        Returns:
            Dictionary with SERP analysis data
        """
        params = {
            "serp_query": {
                "keyword": keyword,
                "locale": "en-US",
                "device": "desktop",
                "engine": "google"
            }
        }
        
        try:
            data = self._make_request("data.serp.analysis.fetch", params)
            return self._process_serp_analysis(data)
        except Exception as e:
            self.logger.error(f"Failed to get SERP analysis: {str(e)}")
            return {}

    def _process_serp_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process SERP analysis response."""
        return {
            'total_results': data.get('total_results', 0),
            'featured_snippet': data.get('featured_snippet', {}),
            'people_also_ask': data.get('people_also_ask', []),
            'related_searches': data.get('related_searches', []),
            'top_ads': data.get('top_ads', []),
            'bottom_ads': data.get('bottom_ads', []),
            'organic_results': data.get('organic_results', [])
        }

    def clear_cache(self) -> None:
        """Clear all cached API responses."""
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")

    def _get_cache_key(self, method: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key for the API request."""
        request_str = f"{method}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(request_str.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if it exists and is not expired."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is expired
                cached_time = datetime.fromisoformat(cached_data['cached_at'])
                if datetime.now() - cached_time < self.cache_duration:
                    self.logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_data['data']
                else:
                    self.logger.debug(f"Cache expired for key: {cache_key}")
                    os.remove(cache_file)
            except Exception as e:
                self.logger.warning(f"Error reading cache: {str(e)}")
                
        return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save API response to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'data': data
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            self.logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")

    def get_competitor_analysis(self, domain: str, competitors: List[str]) -> Dict[str, Any]:
        """
        Perform competitor analysis by comparing metrics.
        
        Args:
            domain: Your domain
            competitors: List of competitor domains
            
        Returns:
            Dictionary with comparative analysis
        """
        try:
            # Get metrics for your domain
            your_metrics = self.get_domain_metrics(domain)
            
            # Get metrics for competitors
            competitor_metrics = {
                competitor: self.get_domain_metrics(competitor)
                for competitor in competitors
            }
            
            return {
                'your_domain': your_metrics,
                'competitors': competitor_metrics
            }
        except Exception as e:
            self.logger.error(f"Failed to perform competitor analysis: {str(e)}")
            return {}

    def get_domain_metrics(self, domain: str) -> Dict[str, Any]:
        """
        Get comprehensive domain metrics using data.site.metrics.fetch.
        
        Args:
            domain: The domain to analyze
            
        Returns:
            Dictionary with domain metrics
        """
        params = {
            "target_query": {
                "query": domain,
                "scope": "domain",
                "locale": "en-US"
            }
        }
        
        try:
            data = self._make_request("data.site.metrics.fetch", params)
            return self._process_domain_metrics(data)
        except Exception as e:
            self.logger.error(f"Failed to get domain metrics: {str(e)}")
            return {}

    def _process_domain_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process domain metrics response."""
        return {
            'domain_authority': data.get('domain_authority', 0),
            'page_authority': data.get('page_authority', 0),
            'spam_score': data.get('spam_score', 0),
            'linking_domains': data.get('linking_domains', 0),
            'rank': data.get('rank', 0),
            'organic_keywords': data.get('organic_keywords', 0),
            'organic_traffic': data.get('organic_traffic', 0)
        }

    def get_content_gap_analysis(self, domain: str, competitor: str) -> Dict[str, Any]:
        """
        Perform content gap analysis between your domain and a competitor.
        
        Args:
            domain: Your domain
            competitor: Competitor domain
            
        Returns:
            Dictionary with content gap analysis
        """
        try:
            # Get ranking keywords for both domains
            your_keywords = self.get_ranking_keywords(domain)
            competitor_keywords = self.get_ranking_keywords(competitor)
            
            # Find keywords competitor ranks for but you don't
            your_keyword_set = {kw['keyword'] for kw in your_keywords}
            gap_keywords = [
                kw for kw in competitor_keywords
                if kw['keyword'] not in your_keyword_set
            ]
            
            return {
                'total_gap_keywords': len(gap_keywords),
                'gap_keywords': sorted(
                    gap_keywords,
                    key=lambda x: x['volume'],
                    reverse=True
                )[:100]  # Return top 100 gap keywords
            }
        except Exception as e:
            self.logger.error(f"Failed to perform content gap analysis: {str(e)}")
            return {}

    def get_backlink_analysis(self, domain: str) -> Dict[str, Any]:
        """
        Get backlink analysis for a domain using data.links.analysis.fetch.
        
        Args:
            domain: The domain to analyze
            
        Returns:
            Dictionary with backlink analysis
        """
        params = {
            "target_query": {
                "query": domain,
                "scope": "domain",
                "locale": "en-US"
            }
        }
        
        try:
            data = self._make_request("data.links.analysis.fetch", params)
            return self._process_backlink_analysis(data)
        except Exception as e:
            self.logger.error(f"Failed to get backlink analysis: {str(e)}")
            return {}

    def _process_backlink_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process backlink analysis response."""
        return {
            'total_backlinks': data.get('total_backlinks', 0),
            'linking_domains': data.get('linking_domains', 0),
            'top_linking_domains': data.get('top_linking_domains', []),
            'top_linking_pages': data.get('top_linking_pages', []),
            'anchor_text_distribution': data.get('anchor_text_distribution', {})
        }

    def find_new_keywords(self, domain: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find new keyword opportunities by analyzing content gaps and related keywords.
        
        Args:
            domain: The domain to analyze
            limit: Maximum number of keywords to return
            
        Returns:
            List of new keyword opportunities with metrics
        """
        try:
            # Get current ranking keywords
            current_keywords = self.get_ranking_keywords(domain, limit=limit)
            current_keyword_set = {kw['keyword'] for kw in current_keywords}
            
            # Get related keywords for top ranking keywords
            new_keywords = []
            for kw in current_keywords[:5]:  # Analyze top 5 keywords
                related = self.get_related_keywords(kw['keyword'], limit=limit)
                for rk in related:
                    if rk['keyword'] not in current_keyword_set:
                        metrics = self.get_keyword_metrics(rk['keyword'])
                        # Add null checks and default values
                        volume = metrics.get('volume', 0) or 0
                        difficulty = metrics.get('difficulty', 100) or 100
                        
                        if (volume >= 100 and difficulty <= 60):
                            rk.update(metrics)
                            rk['opportunity_score'] = self._calculate_opportunity_score(
                                rank=0,  # Not currently ranking
                                volume=volume,
                                difficulty=difficulty
                            )
                            new_keywords.append(rk)
            
            # Sort and return top opportunities
            return sorted(
                new_keywords,
                key=lambda x: x['opportunity_score'],
                reverse=True
            )[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to find new keywords: {str(e)}")
            return []