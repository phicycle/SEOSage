import requests
import base64
import time
from collections import defaultdict
from typing import List, Dict, Any, Set, Optional
import logging
from requests.exceptions import RequestException

class KeywordResearch:
    """
    Performs keyword research and analysis using the Moz API.
    
    Attributes:
        access_id: Moz API access ID
        secret_key: Moz API secret key
        base_url: Moz API base URL
        logger: Logger instance
    """
    
    def __init__(self, access_id: str, secret_key: str) -> None:
        """Initialize keyword research tool with API credentials."""
        self.access_id = access_id
        self.secret_key = secret_key
        self.base_url = "https://api.moz.com/v1/"
        self.logger = logging.getLogger(__name__)
        
        # Common stopwords for keyword clustering
        self._stopwords: Set[str] = {
            'and', 'or', 'the', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from'
        }
        
    def _get_auth_header(self) -> str:
        """Generate authentication header for API requests."""
        try:
            credentials = f"{self.access_id}:{self.secret_key}"
            encoded = base64.b64encode(credentials.encode()).decode()
            return f"Basic {encoded}"
        except Exception as e:
            self.logger.error(f"Error generating auth header: {str(e)}")
            raise ValueError("Invalid API credentials")

    def get_keywords(self, domain: str, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch keyword opportunities for a given domain using Moz API.
        
        Args:
            domain: Target website domain
            limit: Maximum number of keywords to fetch
            
        Returns:
            List of keyword dictionaries with metrics
            
        Raises:
            RequestException: If API request fails
            ValueError: If response is invalid
        """
        headers = {
            'Authorization': self._get_auth_header(),
            'Content-Type': 'application/json'
        }
        
        payload = {
            'target': domain,
            'limit': limit,
            'source': 'domain',
            'sort_by': 'volume',
            'include_serp_features': True,
            'exclude_no_volume': True,
            'search_volume_filter': {
                'min': 100,
                'max': None
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}keywords/explorer",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, dict) or 'keyword_results' not in data:
                raise ValueError("Invalid API response format")
                
            return self._process_keywords(data)
            
        except RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise
        except ValueError as e:
            self.logger.error(f"Invalid response: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise

    def _process_keywords(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process raw API response into structured keyword data.
        
        Args:
            data: Raw API response data
            
        Returns:
            List of processed keyword dictionaries
        """
        keywords = []
        for item in data.get('keyword_results', []):
            try:
                keyword = self._create_keyword_entry(item)
                if keyword:
                    keywords.append(keyword)
            except (ValueError, KeyError) as e:
                self.logger.warning(
                    f"Error processing keyword {item.get('keyword')}: {str(e)}"
                )
                continue
        return keywords

    def _create_keyword_entry(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a structured keyword entry from API data."""
        try:
            return {
                'keyword': item['keyword'],
                'position': int(item.get('serp_position', 0)),
                'search_volume': int(item.get('monthly_volume', 0)),
                'cpc': float(item.get('cpc', 0.0)),
                'difficulty': float(item.get('difficulty', 0.0)),
                'potential': float(item.get('priority', 0.0)),
                'clustering': self._cluster_keyword(item['keyword']),
                'intent': self._determine_search_intent(item['keyword']),
                'serp_features': item.get('serp_features', []),
                'last_updated': time.strftime('%Y-%m-%d')
            }
        except (ValueError, KeyError):
            return None

    def _cluster_keyword(self, keyword: str) -> str:
        """
        Cluster keywords based on semantic similarity.
        
        Args:
            keyword: Keyword to cluster
            
        Returns:
            Normalized keyword cluster identifier
        """
        words = keyword.lower().split()
        words = [w for w in words if w not in self._stopwords]
        
        if len(words) > 1:
            return ' '.join(sorted(words))
        return keyword

    def _determine_search_intent(self, keyword: str) -> str:
        """
        Determine the search intent of a keyword.
        
        Args:
            keyword: Keyword to analyze
            
        Returns:
            One of: 'informational', 'navigational', 'commercial', 'transactional'
        """
        keyword = keyword.lower()
        words = set(keyword.split())
        
        intent_indicators = {
            'informational': {'how', 'what', 'why', 'when', 'where', 'guide', 'tutorial', 'tips'},
            'transactional': {'buy', 'price', 'cost', 'cheap', 'purchase', 'deal', 'order'},
            'commercial': {'best', 'top', 'review', 'comparison', 'vs', 'versus', 'alternative'}
        }
        
        for intent, indicators in intent_indicators.items():
            if any(word in words for word in indicators):
                return intent
                
        return 'navigational'

    def select_target_keywords(
        self, 
        keywords: List[Dict[str, Any]], 
        min_volume: int = 500,
        max_difficulty: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Select the most promising keywords based on multiple factors.
        
        Args:
            keywords: List of keyword dictionaries
            min_volume: Minimum monthly search volume
            max_difficulty: Maximum keyword difficulty (0-1)
            
        Returns:
            List of selected keyword opportunities
        """
        try:
            clusters = self._cluster_keywords(keywords, min_volume, max_difficulty)
            selected = self._select_best_from_clusters(clusters)
            return self._sort_by_opportunity(selected)
            
        except Exception as e:
            self.logger.error(f"Error selecting target keywords: {str(e)}")
            return []

    def _cluster_keywords(
        self, 
        keywords: List[Dict[str, Any]], 
        min_volume: int,
        max_difficulty: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group keywords by intent and clustering."""
        clusters = defaultdict(list)
        
        for kw in keywords:
            if self._meets_criteria(kw, min_volume, max_difficulty):
                cluster_key = f"{kw['clustering']}_{kw['intent']}"
                clusters[cluster_key].append(kw)
                
        return clusters

    def _meets_criteria(
        self, 
        keyword: Dict[str, Any], 
        min_volume: int,
        max_difficulty: float
    ) -> bool:
        """Check if keyword meets selection criteria."""
        return (
            keyword['search_volume'] >= min_volume and
            keyword['difficulty'] <= max_difficulty and
            10 < keyword['position'] <= 30
        )

    def _select_best_from_clusters(
        self, 
        clusters: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Select best keyword from each cluster."""
        selected = []
        for cluster in clusters.values():
            best = max(cluster, key=self._calculate_opportunity_score)
            selected.append(best)
        return selected

    def _calculate_opportunity_score(self, keyword: Dict[str, Any]) -> float:
        """Calculate opportunity score for keyword."""
        return (
            keyword['search_volume'] * 
            keyword['cpc'] * 
            (1 - keyword['difficulty']) * 
            keyword['potential']
        )

    def _sort_by_opportunity(
        self, 
        keywords: List[Dict[str, Any]], 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Sort keywords by opportunity score and return top results."""
        for kw in keywords:
            kw['opportunity_score'] = self._calculate_opportunity_score(kw)
            
        keywords.sort(key=lambda x: x['opportunity_score'], reverse=True)
        return keywords[:limit] 