"""Core SEO optimization functionality."""
from typing import Dict, List, Any, Optional, Set
import re
from bs4 import BeautifulSoup
from collections import defaultdict
import logging
import json
from pathlib import Path
import time
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib

class SEOOptimizer:
    """Enhanced SEO optimization with improved rules and caching."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimizer with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Cache configuration
        self._cache_dir = Path('data/cache/seo')
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_duration = timedelta(days=7)
        
        # Load optimization rules
        self._load_rules()
        
        # Initialize performance metrics
        self._metrics: Dict[str, float] = defaultdict(float)
        
    def _load_rules(self) -> None:
        """Load SEO optimization rules with weights."""
        self.rules = {
            'title': {
                'min_length': 30,
                'max_length': 60,
                'weight': 0.15
            },
            'meta_description': {
                'min_length': 120,
                'max_length': 155,
                'weight': 0.1
            },
            'headers': {
                'h1_count': 1,
                'max_depth': 6,
                'weight': 0.1
            },
            'content': {
                'min_words': 300,
                'keyword_density': {
                    'min': 0.01,
                    'max': 0.03
                },
                'weight': 0.25
            },
            'images': {
                'alt_text': True,
                'max_size_kb': 200,
                'weight': 0.1
            },
            'links': {
                'internal_min': 2,
                'external_min': 1,
                'weight': 0.1
            },
            'mobile': {
                'viewport': True,
                'responsive': True,
                'weight': 0.1
            },
            'performance': {
                'load_time_max': 3.0,
                'weight': 0.1
            }
        }
        
    @lru_cache(maxsize=1000)
    def get_optimization_score(self, url: str, content: str) -> Dict[str, Any]:
        """Get SEO optimization score with caching."""
        cache_key = self._get_cache_key(url, content)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
            
        # Calculate scores
        start_time = time.time()
        scores = self._calculate_scores(content)
        self._metrics['score_calculation_time'] += time.time() - start_time
        
        # Cache results
        self._save_to_cache(cache_key, scores)
        
        return scores
        
    def _calculate_scores(self, content: str) -> Dict[str, Any]:
        """Calculate detailed SEO scores."""
        soup = BeautifulSoup(content, 'html.parser')
        scores = {}
        
        # Title optimization
        title = soup.title.string if soup.title else ''
        scores['title'] = {
            'score': self._get_title_score(title),
            'issues': self._get_title_issues(title)
        }
        
        # Meta description
        meta_desc = soup.find('meta', {'name': 'description'})
        meta_content = meta_desc.get('content', '') if meta_desc else ''
        scores['meta_description'] = {
            'score': self._get_meta_description_score(meta_content),
            'issues': self._get_meta_description_issues(meta_content)
        }
        
        # Headers analysis
        headers = self._analyze_headers(soup)
        scores['headers'] = {
            'score': self._get_headers_score(headers),
            'issues': self._get_headers_issues(headers)
        }
        
        # Content analysis
        content_text = self._extract_content(soup)
        scores['content'] = {
            'score': self._get_content_score(content_text),
            'issues': self._get_content_issues(content_text)
        }
        
        # Image optimization
        images = soup.find_all('img')
        scores['images'] = {
            'score': self._get_images_score(images),
            'issues': self._get_images_issues(images)
        }
        
        # Link analysis
        links = self._analyze_links(soup)
        scores['links'] = {
            'score': self._get_links_score(links),
            'issues': self._get_links_issues(links)
        }
        
        # Mobile optimization
        scores['mobile'] = {
            'score': self._get_mobile_score(soup),
            'issues': self._get_mobile_issues(soup)
        }
        
        # Calculate overall score
        scores['overall'] = self._calculate_overall_score(scores)
        
        return scores
        
    def _get_title_score(self, title: str) -> float:
        """Calculate title optimization score."""
        if not title:
            return 0.0
            
        length = len(title)
        rules = self.rules['title']
        
        if length < rules['min_length']:
            return 0.5
        elif length > rules['max_length']:
            return 0.7
        return 1.0
        
    def _get_title_issues(self, title: str) -> List[str]:
        """Get title optimization issues."""
        issues = []
        rules = self.rules['title']
        
        if not title:
            issues.append("Missing title tag")
            return issues
            
        length = len(title)
        if length < rules['min_length']:
            issues.append(f"Title too short ({length} chars)")
        elif length > rules['max_length']:
            issues.append(f"Title too long ({length} chars)")
            
        return issues
        
    def _get_meta_description_score(self, description: str) -> float:
        """Calculate meta description score."""
        if not description:
            return 0.0
            
        length = len(description)
        rules = self.rules['meta_description']
        
        if length < rules['min_length']:
            return 0.5
        elif length > rules['max_length']:
            return 0.7
        return 1.0
        
    def _get_meta_description_issues(self, description: str) -> List[str]:
        """Get meta description issues."""
        issues = []
        rules = self.rules['meta_description']
        
        if not description:
            issues.append("Missing meta description")
            return issues
            
        length = len(description)
        if length < rules['min_length']:
            issues.append(f"Meta description too short ({length} chars)")
        elif length > rules['max_length']:
            issues.append(f"Meta description too long ({length} chars)")
            
        return issues
        
    def _analyze_headers(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze header structure."""
        headers = {
            'h1_count': len(soup.find_all('h1')),
            'structure': self._get_header_structure(soup),
            'keywords': self._extract_header_keywords(soup)
        }
        return headers
        
    def _get_headers_score(self, headers: Dict[str, Any]) -> float:
        """Calculate headers optimization score."""
        rules = self.rules['headers']
        score = 1.0
        
        # Check H1 count
        if headers['h1_count'] != rules['h1_count']:
            score *= 0.7
            
        # Check header structure
        if not self._is_valid_header_structure(headers['structure']):
            score *= 0.8
            
        return score
        
    def _get_headers_issues(self, headers: Dict[str, Any]) -> List[str]:
        """Get header optimization issues."""
        issues = []
        rules = self.rules['headers']
        
        if headers['h1_count'] == 0:
            issues.append("Missing H1 tag")
        elif headers['h1_count'] > 1:
            issues.append("Multiple H1 tags found")
            
        if not self._is_valid_header_structure(headers['structure']):
            issues.append("Invalid header structure")
            
        return issues
        
    def _get_content_score(self, content: str) -> float:
        """Calculate content optimization score."""
        if not content:
            return 0.0
            
        words = content.split()
        rules = self.rules['content']
        
        if len(words) < rules['min_words']:
            return 0.5
            
        # Check keyword density
        density = self._calculate_keyword_density(content)
        if density < rules['keyword_density']['min']:
            return 0.7
        elif density > rules['keyword_density']['max']:
            return 0.6
            
        return 1.0
        
    def _get_content_issues(self, content: str) -> List[str]:
        """Get content optimization issues."""
        issues = []
        rules = self.rules['content']
        
        if not content:
            issues.append("Missing main content")
            return issues
            
        words = content.split()
        if len(words) < rules['min_words']:
            issues.append(f"Content too short ({len(words)} words)")
            
        density = self._calculate_keyword_density(content)
        if density < rules['keyword_density']['min']:
            issues.append("Keyword density too low")
        elif density > rules['keyword_density']['max']:
            issues.append("Keyword density too high")
            
        return issues
        
    def _calculate_overall_score(self, scores: Dict[str, Any]) -> float:
        """Calculate overall optimization score."""
        total_score = 0.0
        total_weight = 0.0
        
        for category, details in scores.items():
            if category != 'overall':
                weight = self.rules[category]['weight']
                total_score += details['score'] * weight
                total_weight += weight
                
        return total_score / total_weight if total_weight > 0 else 0.0
        
    def _get_cache_key(self, url: str, content: str) -> str:
        """Generate cache key for URL and content."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{url}_{content_hash}"
        
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached optimization results."""
        cache_file = self._cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            data = json.loads(cache_file.read_text())
            cached_time = datetime.fromisoformat(data['timestamp'])
            
            if datetime.now() - cached_time > self._cache_duration:
                return None
                
            return data['scores']
            
        except Exception as e:
            self.logger.error(f"Cache read error: {str(e)}")
            return None
            
    def _save_to_cache(self, cache_key: str, scores: Dict[str, Any]) -> None:
        """Save optimization results to cache."""
        cache_file = self._cache_dir / f"{cache_key}.json"
        
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'scores': scores
            }
            cache_file.write_text(json.dumps(data))
            
        except Exception as e:
            self.logger.error(f"Cache write error: {str(e)}")
            
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Normalize whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        return ' '.join(chunk for chunk in chunks if chunk)
        
    def _calculate_keyword_density(self, content: str) -> float:
        """Calculate keyword density in content."""
        # This is a simplified implementation
        # In practice, you would use the target keywords from the configuration
        words = content.lower().split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
            
        # Count keyword occurrences
        keyword_count = sum(1 for word in words if len(word) > 3)
        
        return keyword_count / total_words
        
    def _get_header_structure(self, soup: BeautifulSoup) -> List[str]:
        """Get hierarchical header structure."""
        headers = []
        for i in range(1, 7):
            for header in soup.find_all(f'h{i}'):
                headers.append(f'h{i}')
        return headers
        
    def _is_valid_header_structure(self, headers: List[str]) -> bool:
        """Check if header structure is valid."""
        if not headers:
            return False
            
        # Check for sequential order
        current_level = 1
        for header in headers:
            level = int(header[1])
            if level > current_level + 1:
                return False
            current_level = level
            
        return True
        
    def _extract_header_keywords(self, soup: BeautifulSoup) -> Set[str]:
        """Extract keywords from headers."""
        keywords = set()
        for i in range(1, 7):
            for header in soup.find_all(f'h{i}'):
                words = header.get_text().lower().split()
                keywords.update(word for word in words if len(word) > 3)
        return keywords
        
    def _analyze_links(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze link structure."""
        links = {
            'internal': [],
            'external': [],
            'broken': []
        }
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('#'):
                continue
                
            if href.startswith('http'):
                links['external'].append(href)
            else:
                links['internal'].append(href)
                
        return links
        
    def _get_links_score(self, links: Dict[str, Any]) -> float:
        """Calculate links optimization score."""
        rules = self.rules['links']
        score = 1.0
        
        if len(links['internal']) < rules['internal_min']:
            score *= 0.8
            
        if len(links['external']) < rules['external_min']:
            score *= 0.9
            
        return score
        
    def _get_links_issues(self, links: Dict[str, Any]) -> List[str]:
        """Get link optimization issues."""
        issues = []
        rules = self.rules['links']
        
        if len(links['internal']) < rules['internal_min']:
            issues.append("Insufficient internal links")
            
        if len(links['external']) < rules['external_min']:
            issues.append("Insufficient external links")
            
        if links['broken']:
            issues.append(f"Found {len(links['broken'])} broken links")
            
        return issues
        
    def _get_images_score(self, images: List[Any]) -> float:
        """Calculate images optimization score."""
        if not images:
            return 1.0
            
        score = 1.0
        for img in images:
            if not img.get('alt'):
                score *= 0.9
                
        return score
        
    def _get_images_issues(self, images: List[Any]) -> List[str]:
        """Get image optimization issues."""
        issues = []
        
        for img in images:
            if not img.get('alt'):
                issues.append(f"Missing alt text for image: {img.get('src', 'unknown')}")
                
        return issues
        
    def _get_mobile_score(self, soup: BeautifulSoup) -> float:
        """Calculate mobile optimization score."""
        score = 1.0
        rules = self.rules['mobile']
        
        # Check viewport
        viewport = soup.find('meta', {'name': 'viewport'})
        if not viewport:
            score *= 0.7
            
        # Check responsive design
        if not self._is_responsive_design(soup):
            score *= 0.8
            
        return score
        
    def _get_mobile_issues(self, soup: BeautifulSoup) -> List[str]:
        """Get mobile optimization issues."""
        issues = []
        
        viewport = soup.find('meta', {'name': 'viewport'})
        if not viewport:
            issues.append("Missing viewport meta tag")
            
        if not self._is_responsive_design(soup):
            issues.append("Non-responsive design detected")
            
        return issues
        
    def _is_responsive_design(self, soup: BeautifulSoup) -> bool:
        """Check if design is responsive."""
        # Check for common responsive design indicators
        viewport = soup.find('meta', {'name': 'viewport'})
        if viewport and 'width=device-width' in viewport.get('content', ''):
            return True
            
        # Check for media queries in style tags
        for style in soup.find_all('style'):
            if '@media' in style.string:
                return True
                
        return False
        
    def clear_cache(self) -> None:
        """Clear optimization cache."""
        try:
            for cache_file in self._cache_dir.glob('*.json'):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return dict(self._metrics)

    async def analyze_technical(self, url: str) -> Dict[str, Any]:
        """Analyze technical SEO aspects."""
        try:
            results = {
                'issues': [],
                'metrics': {},
                'recommendations': []
            }
            
            # Analyze various technical aspects
            results['issues'].extend(self._get_technical_issues(url))
            results['metrics'].update(self._get_technical_metrics(url))
            results['recommendations'].extend(
                self._generate_technical_recommendations(url)
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {str(e)}")
            return {'error': str(e)}

    async def optimize_meta_tags(
        self,
        content: str,
        keyword_data: Dict[str, Any],
        content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize meta tags with context."""
        try:
            return {
                'meta_tags': self._generate_optimized_meta_tags(
                    content,
                    keyword_data
                ),
                'recommendations': self._get_meta_optimization_recommendations(
                    content_analysis
                ),
                'optimization_score': self._calculate_meta_optimization_score(
                    content,
                    keyword_data
                )
            }
        except Exception as e:
            self.logger.error(f"Error optimizing meta tags: {str(e)}")
            return {'error': str(e)}

    async def analyze_performance(self, url: str) -> Dict[str, Any]:
        """Analyze page performance."""
        try:
            return {
                'performance_metrics': self._get_performance_metrics(url),
                'optimization_suggestions': self._get_performance_suggestions(url),
                'critical_issues': self._get_critical_performance_issues(url)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {str(e)}")
            return {'error': str(e)}

    async def generate_recommendations(
        self,
        content: str,
        technical_data: Dict[str, Any],
        content_data: Dict[str, Any],
        keyword_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive SEO recommendations."""
        try:
            return {
                'technical': self._get_technical_recommendations(technical_data),
                'content': self._get_content_recommendations(content_data),
                'meta': self._get_meta_recommendations(content, keyword_data),
                'performance': self._get_performance_recommendations(technical_data),
                'priority_actions': self._get_priority_actions(
                    technical_data,
                    content_data,
                    keyword_data
                )
            }
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {'error': str(e)}