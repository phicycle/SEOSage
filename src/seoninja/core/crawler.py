"""Core website crawling functionality."""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import urllib.error
from collections import defaultdict, deque
import logging
from typing import Dict, List, Optional, Any, Set, Generator
import logging.handlers
import os
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import aiohttp
import asyncio
from aiohttp import ClientTimeout, ClientSession
import urllib.request
import json
import hashlib
from pathlib import Path
import math

class MemoryEfficientSet:
    """Memory efficient set implementation using bloom filter."""
    
    def __init__(self, expected_items: int = 10000, error_rate: float = 0.01):
        self.size = self._get_optimal_size(expected_items, error_rate)
        self.hash_funcs = self._get_optimal_hashes(expected_items, self.size)
        self.bit_array = [0] * self.size
        self._added_items = 0
        
    def add(self, item: str) -> None:
        """Add an item to the set."""
        for seed in range(self.hash_funcs):
            index = self._get_hash_index(item, seed)
            self.bit_array[index] = 1
        self._added_items += 1
        
    def __contains__(self, item: str) -> bool:
        """Check if an item might be in the set."""
        return all(
            self.bit_array[self._get_hash_index(item, seed)] == 1
            for seed in range(self.hash_funcs)
        )
        
    def _get_hash_index(self, item: str, seed: int) -> int:
        """Get hash index for an item."""
        return hash((item, seed)) % self.size
        
    @staticmethod
    def _get_optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        return int(-n * math.log(p) / (math.log(2) ** 2))
        
    @staticmethod
    def _get_optimal_hashes(n: int, m: int) -> int:
        """Calculate optimal number of hash functions."""
        return int((m / n) * math.log(2))

class WebsiteCrawler:
    """Enhanced web crawler with improved performance and memory management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize crawler with enhanced configuration."""
        self.config = config or {}
        self.domain = self.config.get('domain', '')
        self.max_pages = self.config.get('max_pages', 1000)
        self.max_depth = self.config.get('max_depth', 5)
        
        # Use memory efficient set for visited URLs
        self.visited_urls = MemoryEfficientSet(
            expected_items=self.max_pages * 2
        )
        
        # Use deque for URL queue with maxlen
        self.url_queue = deque(maxlen=self.max_pages * 2)
        
        # Efficient content storage with LRU cache
        self.content_data = {}
        self._content_cache_size = 100
        
        # Enhanced session configuration
        self.session_config = {
            'timeout': ClientTimeout(total=30),
            'headers': {
                'User-Agent': self.config.get(
                    'user_agent',
                    'SEONinja/1.0'
                )
            },
            'max_redirects': 5
        }
        
        # Rate limiting
        self.rate_limit = self.config.get('rate_limit', 1.0)
        self.last_request_time = 0
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Initialize other components
        self.logger = logging.getLogger(__name__)
        self.robot_parser = None
        self.sitemap_urls = set()
        self.session = None
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
    async def _init_session(self) -> None:
        """Initialize session with retry logic."""
        for attempt in range(self.max_retries):
            try:
                if self.session is None or self.session.closed:
                    self.session = ClientSession(**self.session_config)
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                
    async def _process_sitemaps(self) -> None:
        """Process sitemaps with batching and retries."""
        try:
            sitemap_urls = await self._find_sitemaps()
            if not sitemap_urls:
                self.logger.info("No sitemaps found")
                return
                
            # Process sitemaps in batches
            batch_size = 10
            for i in range(0, len(sitemap_urls), batch_size):
                batch = sitemap_urls[i:i + batch_size]
                
                # Process batch with retries
                for attempt in range(self.max_retries):
                    try:
                        tasks = [self._process_sitemap(url) for url in batch]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Handle exceptions
                        for url, result in zip(batch, results):
                            if isinstance(result, Exception):
                                self.logger.error(
                                    f"Failed to process sitemap {url}: {str(result)}"
                                )
                        break
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            self.logger.error(
                                f"Failed to process sitemap batch after {self.max_retries} attempts"
                            )
                        else:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
        except Exception as e:
            self.logger.error(f"Error processing sitemaps: {str(e)}")
        finally:
            self.logger.debug("Finished processing sitemaps")
            
    async def _crawl_url(self, url: str, depth: int = 0) -> None:
        """Crawl a URL with improved error handling and memory management."""
        if depth > self.max_depth or url in self.visited_urls:
            return
            
        try:
            # Rate limiting
            await self._respect_rate_limit()
            
            # Use semaphore for concurrent requests
            async with self.semaphore:
                # Fetch and process with retries
                for attempt in range(self.max_retries):
                    try:
                        async with self.session.get(url) as response:
                            if response.status == 200:
                                content = await response.text()
                                
                                # Process content
                                await self._process_page(url, content)
                                
                                # Extract and queue new URLs
                                new_urls = self._extract_urls(content, url)
                                for new_url in new_urls:
                                    if len(self.url_queue) < self.max_pages:
                                        self.url_queue.append((new_url, depth + 1))
                                        
                                break
                            else:
                                self.logger.warning(
                                    f"Failed to fetch {url}, status: {response.status}"
                                )
                                
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            self.logger.error(f"Failed to crawl {url}: {str(e)}")
                        else:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {str(e)}")
            
        finally:
            self.visited_urls.add(url)
            
    async def _respect_rate_limit(self) -> None:
        """Respect rate limiting with sleep."""
        if self.rate_limit > 0:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                await asyncio.sleep(self.rate_limit - time_since_last)
            self.last_request_time = time.time()
            
    def _manage_content_cache(self) -> None:
        """Manage content cache size."""
        if len(self.content_data) > self._content_cache_size:
            # Remove oldest entries
            sorted_items = sorted(
                self.content_data.items(),
                key=lambda x: x[1].get('timestamp', 0)
            )
            self.content_data = dict(
                sorted_items[-self._content_cache_size:]
            )
            
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain format."""
        # Remove any trailing slashes
        domain = domain.rstrip('/')
        if not domain.startswith(('http://', 'https://')):
            return f'https://{domain}'
        return domain
        
    def _setup_robot_parser(self) -> Optional[RobotFileParser]:
        """Setup robots.txt parser with error handling."""
        parser = RobotFileParser()
        try:
            parser.set_url(f"{self.domain}/robots.txt")
            # Try to read robots.txt with timeout
            with urllib.request.urlopen(f"{self.domain}/robots.txt", timeout=10) as response:
                if response.status == 200:
                    parser.read()
                    self.logger.debug(f"Found and parsed robots.txt at {self.domain}")
                    return parser
            self.logger.debug(f"No robots.txt found at {self.domain}")
            return None
        except (urllib.error.URLError, ValueError, urllib.error.HTTPError) as e:
            self.logger.info(f"No robots.txt found at {self.domain}, allowing all URLs")
            return None
            
    def can_fetch(self, url: str) -> bool:
        """
        Check if URL can be fetched according to robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if URL can be fetched
        """
        # If no robot parser exists (no robots.txt found), allow all URLs
        if not self.robot_parser:
            return True
        try:
            return self.robot_parser.can_fetch("*", url)
        except Exception as e:
            self.logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
            return True
            
    def _extract_page_data(
        self, 
        url: str, 
        soup: BeautifulSoup, 
        response_time: float
    ) -> Dict[str, Any]:
        """
        Extract structured data from a page with better error handling.
        """
        try:
            page_data = {
                'url': url,
                'title': self._get_title(soup),
                'meta_description': self._get_meta_description(soup),
                'headers': self._get_headers(soup),
                'content': self._get_content(soup),
                'links': [],  # Populated during crawl
                'canonical': self._get_canonical(soup),
                'images': self._get_images(soup),
                'response_time': response_time
            }
            
            # Validate extracted data
            if not page_data['content']:
                self.logger.warning(f"Empty content extracted from {url}")
            
            return page_data
        except Exception as e:
            self.logger.error(f"Error extracting data from {url}: {str(e)}")
            return self._get_empty_page_data(url, response_time)
            
    def _get_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        return soup.title.string.strip() if soup.title else ''
        
    def _get_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description."""
        meta = soup.find('meta', attrs={'name': 'description'})
        return meta['content'].strip() if meta and 'content' in meta.attrs else ''
        
    def _get_headers(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract header hierarchy."""
        return {
            f'h{i}': [h.text.strip() for h in soup.find_all(f'h{i}')]
            for i in range(1, 4)
        }
        
    def _get_content(self, soup: BeautifulSoup) -> str:
        """Extract and clean page content with better handling."""
        try:
            # Remove script and style elements
            for element in soup(['script', 'style', 'noscript', 'iframe']):
                element.decompose()
            
            # Get text and clean it
            text = soup.get_text(separator=' ')
            # Remove extra whitespace and normalize
            return ' '.join(text.split())
        except Exception as e:
            self.logger.warning(f"Error extracting content: {str(e)}")
            return ''
        
    def _get_canonical(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract canonical URL."""
        canonical = soup.find('link', {'rel': 'canonical'})
        return canonical['href'] if canonical else None
        
    def _get_images(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract image data."""
        return [
            {'src': img.get('src', ''), 'alt': img.get('alt', '')}
            for img in soup.find_all('img')
        ]
            
    def _get_empty_page_data(self, url: str, response_time: float) -> Dict[str, Any]:
        """Return empty page data structure for error cases."""
        return {
            'url': url,
            'title': '',
            'meta_description': '',
            'headers': {f'h{i}': [] for i in range(1, 4)},
            'content': '',
            'links': [],
            'canonical': None,
            'images': [],
            'response_time': response_time
        }
        
    async def _find_sitemaps(self) -> List[str]:
        """Find sitemap URLs from robots.txt and common locations."""
        sitemap_urls = set()
        
        # Check robots.txt for sitemap
        if self.robot_parser:
            self.logger.info("Checking robots.txt for sitemaps")
            try:
                sitemaps = self.robot_parser.site_maps()
                if sitemaps:
                    sitemap_urls.update(sitemaps)
            except (AttributeError, TypeError) as e:
                self.logger.warning(f"Could not get sitemaps from robots.txt: {str(e)}")
        
        # Check common sitemap locations
        common_locations = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemap/',
            '/sitemap/sitemap.xml',
            '/wp-sitemap.xml',
            '/sitemap1.xml'
        ]
        
        for location in common_locations:
            url = urljoin(self.domain, location)
            try:
                response = await self._make_request(url)
                content_type = response.headers.get('Content-Type', '').lower()
                if 'xml' in content_type:
                    sitemap_urls.add(url)
                    self.logger.info(f"Found sitemap at: {url}")
            except Exception as e:
                self.logger.debug(f"Could not access sitemap at {url}: {str(e)}")
                continue
        
        return list(sitemap_urls)
        
    async def _make_request(self, url: str, retries: int = 3) -> aiohttp.ClientResponse:
        """Make HTTP request with retries and rate limiting."""
        if not self.session:
            await self._init_session()
            
        async with self.semaphore:  # Rate limit requests
            for attempt in range(retries):
                try:
                    response = await self.session.get(url, allow_redirects=True)
                    if response.status == 429:  # Too Many Requests
                        wait_time = int(response.headers.get('Retry-After', 60))
                        self.logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    response.raise_for_status()
                    return response
                except Exception as e:
                    if attempt == retries - 1:  # Last attempt
                        self.logger.error(f"Failed to fetch {url} after {retries} attempts: {str(e)}")
                        raise
                    wait_time = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()
        
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"
        
    def _save_to_cache(self, url: str, data: Dict[str, Any]) -> None:
        """Save data to cache."""
        try:
            cache_key = self._get_cache_key(url)
            cache_path = self._get_cache_path(cache_key)
            
            cache_data = {
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
            self.logger.debug(f"Cached data for {url}")
        except Exception as e:
            self.logger.warning(f"Failed to cache data for {url}: {str(e)}")
            
    def _get_from_cache(self, url: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if fresh."""
        try:
            cache_key = self._get_cache_key(url)
            cache_path = self._get_cache_path(cache_key)
            
            if not cache_path.exists():
                return None
                
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is fresh
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > self.cache_duration:
                self.logger.debug(f"Cache expired for {url}")
                return None
                
            self.logger.debug(f"Using cached data for {url}")
            return cache_data['data']
            
        except Exception as e:
            self.logger.warning(f"Failed to read cache for {url}: {str(e)}")
            return None
            
    async def _parse_sitemap(self, sitemap_url: str) -> Set[str]:
        """Parse sitemap XML and extract URLs."""
        urls = set()
        try:
            response = await self._make_request(sitemap_url)
            content = await response.text()
            
            # Check if it's a sitemap index
            if '<sitemapindex' in content.lower():
                # Parse sitemap index
                root = ET.fromstring(content)
                for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    sub_urls = await self._parse_sitemap(sitemap.text)
                    urls.update(sub_urls)
            else:
                # Parse regular sitemap
                root = ET.fromstring(content)
                for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    urls.add(url.text)
                    
        except Exception as e:
            self.logger.error(f"Error parsing sitemap {sitemap_url}: {str(e)}")
            
        return urls
        
    async def crawl_site(self) -> None:
        """
        Crawl the website starting from the domain.
        
        This method:
        1. Initializes the HTTP session
        2. Processes sitemaps if available
        3. Crawls the main domain and discovered URLs
        4. Cleans up resources
        """
        try:
            # Initialize session
            await self._init_session()
            
            # Start with processing sitemaps
            await self._process_sitemaps()
            
            # Start crawling from domain
            await self._crawl_url(self.domain)
            
            # Process discovered URLs from sitemap that weren't crawled
            sitemap_tasks = []
            for url in self.sitemap_urls:
                if url not in self.visited_urls and len(self.visited_urls) < self.max_pages:
                    sitemap_tasks.append(asyncio.create_task(self._crawl_url(url)))
                    
            if sitemap_tasks:
                await asyncio.gather(*sitemap_tasks)
                
        except Exception as e:
            self.logger.error(f"Error during crawl: {str(e)}")
        finally:
            # Cleanup
            if self.session:
                await self.session.close()
                self.session = None
                
    async def _process_sitemap(self, sitemap_url: str) -> None:
        """Process a single sitemap."""
        try:
            urls = await self._parse_sitemap(sitemap_url)
            self.sitemap_urls.update(urls)
            self.logger.info(f"Found {len(urls)} URLs in sitemap {sitemap_url}")
        except Exception as e:
            self.logger.error(f"Error processing sitemap {sitemap_url}: {str(e)}")
            
    def _parse_sitemap_content(self, content: str) -> Set[str]:
        """Parse sitemap content and extract URLs."""
        urls = set()
        try:
            root = ET.fromstring(content)
            
            # Handle sitemap index
            if root.tag.endswith('sitemapindex'):
                for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    urls.add(sitemap.text)
            # Handle regular sitemap
            else:
                for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    urls.add(url.text)
                    
        except ET.ParseError as e:
            self.logger.error(f"Error parsing sitemap XML: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing sitemap: {str(e)}")
            
        return urls
        
    async def _get_page_links(self, url: str) -> Set[str]:
        """Extract links from a page."""
        links = set()
        try:
            response = await self._make_request(url)
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                if self._is_valid_internal_link(absolute_url):
                    links.add(absolute_url)
                    
        except Exception as e:
            self.logger.error(f"Error getting links from {url}: {str(e)}")
            
        return links
        
    def analyze_content(self) -> Dict[str, any]:
        """Analyze crawled content."""
        try:
            analysis = {
                'total_pages': len(self.content_data),
                'avg_response_time': sum(page['response_time'] for page in self.content_data.values()) / len(self.content_data) if self.content_data else 0,
                'pages_with_missing_meta': len([page for page in self.content_data.values() if not page['meta_description']]),
                'pages_with_missing_title': len([page for page in self.content_data.values() if not page['title']]),
                'total_internal_links': sum(len(page['links']) for page in self.content_data.values()),
                'total_images': sum(len(page['images']) for page in self.content_data.values()),
                'pages': self.content_data
            }
            return analysis
        except Exception as e:
            self.logger.error(f"Error analyzing content: {str(e)}")
            return self._get_empty_analysis()
            
    def _get_empty_analysis(self) -> Dict[str, any]:
        """Return empty analysis structure."""
        return {
            'total_pages': 0,
            'avg_response_time': 0,
            'pages_with_missing_meta': 0,
            'pages_with_missing_title': 0,
            'total_internal_links': 0,
            'total_images': 0,
            'pages': {}
        }
        
    def crawl(self, url: Optional[str] = None) -> None:
        """
        Synchronous wrapper for crawling.
        
        Args:
            url: Optional URL to start crawling from. If not provided,
                 uses the domain URL.
        """
        try:
            start_url = url if url else self.domain
            if not self._is_valid_internal_link(start_url):
                raise ValueError(f"Invalid start URL: {start_url}")
                
            # Run the async crawl
            asyncio.run(self.crawl_site())
            
            self.logger.info(f"Crawl completed. Processed {len(self.visited_urls)} pages")
            
        except Exception as e:
            self.logger.error(f"Crawl failed: {str(e)}")
            raise
            
    def _is_valid_internal_link(self, url: str) -> bool:
        """Check if URL is valid and internal to the domain."""
        try:
            parsed_url = urlparse(url)
            parsed_domain = urlparse(self.domain)
            
            # Check if URL is internal
            is_internal = parsed_url.netloc == parsed_domain.netloc
            
            # Check if URL is HTTP/HTTPS
            is_http = parsed_url.scheme in ('http', 'https')
            
            return is_internal and is_http
            
        except Exception:
            return False
            
    def clear_cache(self) -> None:
        """Clear the crawler cache."""
        try:
            for cache_file in self.cache_dir.glob('*.json'):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}") 