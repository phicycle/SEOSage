import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import urllib.error
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Any, Set, Generator
import logging.handlers
import os
from datetime import datetime
import xml.etree.ElementTree as ET
import aiohttp
import asyncio
from aiohttp import ClientTimeout
import urllib.request

class WebsiteCrawler:
    """
    Crawls websites to analyze content and structure.
    
    Attributes:
        domain: Website domain to crawl
        visited_urls: Set of already crawled URLs
        content_data: Dictionary storing page data
        robot_parser: Robot.txt parser instance
        logger: Logger instance
        sitemap_urls: Set of URLs found in sitemaps
        session: aiohttp ClientSession instance
        semaphore: asyncio Semaphore for limiting concurrent requests
    """
    
    def __init__(self, domain: str) -> None:
        """
        Initialize crawler with domain and setup logging.
        
        Args:
            domain: Website domain to crawl
            
        Raises:
            ValueError: If domain is empty
        """
        # Setup logging if not already configured
        if not logging.getLogger().handlers:
            setup_logging()
        
        self.logger = logging.getLogger(__name__)
        
        if not domain:
            self.logger.error("Domain cannot be empty")
            raise ValueError("Domain cannot be empty")
            
        self.domain = self._normalize_domain(domain)
        self.visited_urls: Set[str] = set()
        self.content_data: Dict[str, Dict[str, Any]] = {}
        self.robot_parser = self._setup_robot_parser()
        self.sitemap_urls: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        self.timeout = ClientTimeout(total=30)
        self.max_pages = 100
        
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
        
    async def _init_session(self):
        """Initialize aiohttp session with TCP connector settings"""
        if self.session is not None:
            await self.session.close()
            self.session = None
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=10)
        connector = aiohttp.TCPConnector(
            force_close=True,
            enable_cleanup_closed=True,
            limit=10,  # Limit concurrent connections
            ttl_dns_cache=300,  # Cache DNS results for 5 minutes
            ssl=False  # Handle SSL in request
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; SEOAgentBot/1.0; +http://www.example.com/bot.html)',
                'Accept': 'text/html,application/xml,text/xml,application/xhtml+xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
        )

    async def _make_request(self, url: str, retries: int = 3) -> aiohttp.ClientResponse:
        """
        Make async HTTP request with retries and better error handling.
        """
        if not self.session:
            await self._init_session()
        
        for attempt in range(retries):
            try:
                async with self.semaphore:
                    async with self.session.get(url, ssl=True, timeout=self.timeout) as response:
                        # Ensure we read the content while the response is open
                        content = await response.read()
                        if not content:
                            raise ValueError(f"Empty response from {url}")
                        return response
            except aiohttp.ClientSSLError:
                self.logger.warning(f"SSL verification failed for {url}, retrying without verification")
                async with self.session.get(url, ssl=False, timeout=self.timeout) as response:
                    content = await response.read()
                    if not content:
                        raise ValueError(f"Empty response from {url}")
                    return response
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                if attempt == retries - 1:  # Last attempt
                    self.logger.error(f"Failed to fetch {url} after {retries} attempts: {str(e)}")
                    raise
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}, retrying...")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                
                # Reinitialize session if needed
                if isinstance(e, aiohttp.ClientOSError):
                    await self._init_session()

    async def _crawl_url(self, url: str) -> None:
        """Crawl a single URL with improved error handling and logging."""
        url = url.rstrip('/')
        
        if url in self.visited_urls:
            return
            
        if not self.can_fetch(url):
            self.logger.debug(f"Skipping {url} - blocked by robots.txt")
            return
            
        try:
            self.logger.debug(f"Starting crawl of URL: {url}")
            start_time = time.time()
            
            response = await self._make_request(url)
            
            # Validate response status
            if response.status != 200:
                self.logger.warning(f"Skipping {url} - non-200 status: {response.status}")
                return
            
            # Handle redirects
            if response.history:
                url = str(response.url).rstrip('/')
                if url in self.visited_urls:
                    return
            
            # Verify content type
            content_type = response.headers.get('Content-Type', '').lower()
            if not any(html_type in content_type for html_type in ['text/html', 'application/xhtml+xml']):
                self.logger.debug(f"Skipping non-HTML content at {url}")
                return
            
            # Process HTML content
            html_content = await response.text()
            if not html_content:
                self.logger.warning(f"Empty content at {url}")
                return
            
            # Debug: Log first 200 characters of HTML
            self.logger.debug(f"HTML content preview for {url}: {html_content[:200]}...")
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Debug: Log soup status
            self.logger.debug(f"BeautifulSoup parsed content for {url}")
            
            # Extract and store page data
            page_data = self._extract_page_data(url, soup, time.time() - start_time)
            
            # Debug: Log extracted data
            self.logger.debug(f"Extracted data for {url}: {page_data}")
            
            # Store the data in content_data
            self.content_data[url] = page_data
            self.visited_urls.add(url)
            
            self.logger.info(f"Successfully processed {url}")
            self.logger.info(f"Content data size after processing {url}: {len(self.content_data)}")
            
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {str(e)}", exc_info=True)
            # Store empty data for failed URLs
            self.content_data[url] = self._get_empty_page_data(url, 0)

    async def _parse_sitemap(self, sitemap_url: str) -> Set[str]:
        """Parse sitemap asynchronously."""
        urls = set()
        try:
            response = await self._make_request(sitemap_url)
            content = await response.text()
            
            # Debug logging
            self.logger.debug(f"Sitemap content type: {response.headers.get('Content-Type')}")
            self.logger.debug(f"Sitemap content length: {len(content)}")
            
            try:
                root = ET.fromstring(content)
            except ET.ParseError as e:
                self.logger.error(f"XML parsing error for {sitemap_url}: {str(e)}")
                self.logger.debug(f"Content preview: {content[:200]}")
                return urls
            
            # Try different namespace patterns
            namespaces = [
                {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'},
                {},  # No namespace
            ]
            
            for ns in namespaces:
                try:
                    if root.tag.endswith('sitemapindex'):
                        # Handle sitemap index
                        loc_elements = root.findall('.//ns:loc', namespaces=ns) if ns else root.findall('.//loc')
                        if loc_elements:
                            tasks = []
                            for sitemap in loc_elements:
                                if sitemap.text:
                                    tasks.append(self._parse_sitemap(sitemap.text.strip()))
                            if tasks:
                                results = await asyncio.gather(*tasks, return_exceptions=True)
                                for result in results:
                                    if isinstance(result, Exception):
                                        self.logger.error(f"Error in nested sitemap: {str(result)}")
                                    else:
                                        urls.update(result)
                            break  # Found and processed elements, exit namespace loop
                    else:
                        # Handle regular sitemap
                        loc_elements = root.findall('.//ns:loc', namespaces=ns) if ns else root.findall('.//loc')
                        if loc_elements:
                            for url in loc_elements:
                                if url.text:
                                    urls.add(url.text.strip())
                            break  # Found and processed elements, exit namespace loop
                        
                except Exception as e:
                    self.logger.debug(f"Error with namespace {ns}: {str(e)}, trying next pattern")
                    continue
                
        except Exception as e:
            self.logger.error(f"Error parsing sitemap {sitemap_url}: {str(e)}")
            
        return urls

    async def crawl_site(self) -> None:
        """Main crawling function."""
        self.logger.info("Step 1: Initializing session")
        await self._init_session()
        
        try:
            self.logger.info(f"Step 2: Starting crawl of domain: {self.domain}")
            
            # Step 3: Process sitemaps
            self.logger.info("Step 3: Processing sitemaps")
            await self._process_sitemaps()
            
            # Step 4: Crawl sitemap URLs
            self.logger.info("Step 4: Crawling sitemap URLs")
            if self.sitemap_urls:
                valid_urls = [url for url in self.sitemap_urls 
                            if url.startswith(self.domain)]
                self.logger.info(f"Starting to crawl {len(valid_urls)} valid URLs from sitemap")
                
                # Process URLs in batches
                batch_size = 5
                for i in range(0, len(valid_urls), batch_size):
                    batch = valid_urls[i:i + batch_size]
                    # Create tasks for the batch
                    tasks = [self._crawl_url(url) for url in batch]
                    # Wait for all tasks to complete
                    await asyncio.gather(*tasks)
                    self.logger.info(f"Processed {min(i + batch_size, len(valid_urls))}/{len(valid_urls)} URLs")
                    self.logger.info(f"Current content data size: {len(self.content_data)}")
            
            # Step 5: Recursive crawl if needed
            self.logger.info("Step 5: Starting recursive crawl from main domain")
            if not self.content_data:
                await self._crawl_url(self.domain)
                
                # Crawl discovered links
                discovered_urls = await self._get_page_links(self.domain)
                self.logger.info(f"Discovered links: {discovered_urls}")
                
                valid_urls = [url for url in discovered_urls 
                            if url.startswith(self.domain)][:self.max_pages]
                
                self.logger.info(f"Valid URLs to crawl: {valid_urls}")
                
                # Process discovered URLs in batches
                for i in range(0, len(valid_urls), 5):
                    batch = valid_urls[i:i + 5]
                    tasks = [self._crawl_url(url) for url in batch]
                    await asyncio.gather(*tasks)
                    self.logger.info(f"Recursively processed {min(i + 5, len(valid_urls))}/{len(valid_urls)} URLs")
                    self.logger.info(f"Current content data size: {len(self.content_data)}")
            
            if not self.content_data:
                raise ValueError("No pages were successfully crawled")
                
            self.logger.info(f"Crawl completed. Total pages crawled: {len(self.content_data)}")
            
        except Exception as e:
            self.logger.error(f"Error in crawl_site: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.close()
                
    async def _process_sitemaps(self) -> None:
        """Process robots.txt and sitemaps."""
        self.logger.info("Checking robots.txt for sitemaps")
        robots_url = f"{self.domain}/robots.txt"
        
        try:
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    sitemap_urls = [line.split(': ')[1].strip() 
                                  for line in robots_content.split('\n') 
                                  if line.lower().startswith('sitemap:')]
                    
                    for sitemap_url in sitemap_urls:
                        self.logger.info(f"Found sitemap at: {sitemap_url}")
                        await self._process_sitemap(sitemap_url)
                        
            # Check default sitemap locations if none found
            if not self.sitemap_urls:
                default_sitemaps = [
                    f"{self.domain}/sitemap.xml",
                    f"{self.domain}/sitemap_index.xml"
                ]
                for sitemap_url in default_sitemaps:
                    await self._process_sitemap(sitemap_url)
                    
            self.logger.info(f"Found {len(self.sitemap_urls)} URLs in sitemaps")
            
        except Exception as e:
            self.logger.warning(f"Error processing robots.txt: {str(e)}")
            
    async def _process_sitemap(self, sitemap_url: str) -> None:
        """Process a sitemap XML file."""
        try:
            async with self.session.get(sitemap_url) as response:
                if response.status == 200:
                    sitemap_content = await response.text()
                    
                    # Try parsing with different XML patterns
                    try:
                        urls = self._parse_sitemap_content(sitemap_content)
                        self.sitemap_urls.update(urls)
                    except Exception as e:
                        self.logger.warning(f"Error parsing sitemap {sitemap_url}: {str(e)}")
                        
        except Exception as e:
            self.logger.warning(f"Error fetching sitemap {sitemap_url}: {str(e)}")
            
    def _parse_sitemap_content(self, content: str) -> Set[str]:
        """Parse sitemap content and extract URLs."""
        urls = set()
        try:
            root = ET.fromstring(content)
            
            # Handle both regular sitemaps and sitemap indexes
            namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            # Look for URLs in regular sitemap
            for url in root.findall('.//sm:url/sm:loc', namespaces):
                urls.add(url.text)
                
            # Look for URLs in sitemap index
            for sitemap in root.findall('.//sm:sitemap/sm:loc', namespaces):
                urls.add(sitemap.text)
                
        except ET.ParseError:
            # Try parsing without namespace
            try:
                root = ET.fromstring(content)
                urls.update(url.text for url in root.findall('.//loc'))
            except ET.ParseError as e:
                self.logger.error(f"Failed to parse sitemap XML: {str(e)}")
                
        return urls
        
    async def _get_page_links(self, url: str) -> Set[str]:
        """Get all links from a page."""
        links = set()
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        absolute_url = urljoin(url, href)
                        if absolute_url.startswith(self.domain):
                            links.add(absolute_url)
                            
        except Exception as e:
            self.logger.error(f"Error getting links from {url}: {str(e)}")
            
        return links
        
    def analyze_content(self) -> Dict[str, any]:
        """Analyze crawled content."""
        if not self.content_data:
            return self._get_empty_analysis()
            
        total_words = 0
        for url, page_data in self.content_data.items():
            content = page_data.get('content', '')
            if isinstance(content, str):
                total_words += len(content.split())
        
        analysis = {
            'pages_crawled': len(self.content_data),
            'total_words': total_words,
            'avg_page_length': total_words / len(self.content_data) if self.content_data else 0,
            'crawl_timestamp': datetime.now().isoformat(),
            'urls_crawled': list(self.content_data.keys())
        }
        
        return analysis
        
    def _get_empty_analysis(self) -> Dict[str, any]:
        """Return empty analysis structure."""
        return {
            'pages_crawled': 0,
            'total_words': 0,
            'avg_page_length': 0,
            'crawl_timestamp': datetime.now().isoformat(),
            'error': 'No content crawled'
        }

    def crawl(self, url: Optional[str] = None) -> None:
        """
        Synchronous wrapper for async crawl_site.
        
        Args:
            url: Optional URL to crawl (defaults to domain)
        """
        try:
            # Try with default event loop
            asyncio.run(self.crawl_site())
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Handle case where event loop is closed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.crawl_site())
                finally:
                    loop.close()
            else:
                # Re-raise other runtime errors
                raise
        except Exception as e:
            self.logger.error(f"Error during crawl: {str(e)}")
            raise
        
    def _is_valid_internal_link(self, url: str) -> bool:
        """Check if URL is a valid internal link to crawl."""
        try:
            # Remove any trailing slashes for consistent comparison
            url = url.rstrip('/')
            # Extract domain from URL for comparison
            parsed_url = urlparse(url)
            parsed_domain = urlparse(self.domain)
            
            return (
                parsed_url.netloc == parsed_domain.netloc and
                url not in self.visited_urls and
                url.startswith(('http://', 'https://')) and
                not url.endswith(('.pdf', '.jpg', '.png', '.gif', '.css', '.js'))  # Skip non-HTML resources
            )
        except Exception as e:
            self.logger.warning(f"Error validating URL {url}: {str(e)}")
            return False

def setup_logging(log_level=logging.INFO) -> None:
    """
    Configure logging for the crawler with both file and console handlers.
    
    Args:
        log_level: Logging level (default: logging.INFO)
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (rotating file handler to manage log size)
    log_file = os.path.join(log_dir, f'crawler_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Remove existing handlers if any
    logger.handlers.clear()
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler) 