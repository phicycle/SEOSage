"""Web Crawler Agent implementation."""
from typing import Dict, Any, List
from langchain.agents import Tool
from ..base.agent import BaseAgent
from ...core.crawler import WebsiteCrawler
from ...utils.cache import Cache

class CrawlerAgent(BaseAgent):
    """Agent specialized in web crawling and site analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Crawler agent with its specific tools."""
        super().__init__("crawler")
        self.config = config or {}
        self.crawler = WebsiteCrawler(
            config={
                'domain': self.config.get('domain'),
                'max_pages': self.config.get('max_pages', 1000),
                'max_depth': self.config.get('max_depth', 5),
                'rate_limit': self.config.get('rate_limit', 1.0),
                'user_agent': 'SEONinja/1.0'  # Set default user agent here
            }
        )
        self.cache = Cache('crawler')
        self._setup_tools()
        
    def _setup_tools(self) -> List[Tool]:
        """Set up crawler specific tools."""
        return [
            Tool(
                name="crawl_site",
                func=self._crawl_site,
                description="Crawls a website and extracts content and structure"
            ),
            Tool(
                name="analyze_structure",
                func=self._analyze_structure,
                description="Analyzes site structure and navigation"
            ),
            Tool(
                name="extract_content",
                func=self._extract_content,
                description="Extracts and analyzes page content"
            ),
            Tool(
                name="check_links",
                func=self._check_links,
                description="Checks internal and external links"
            )
        ]
        
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a crawling task with enhanced context awareness."""
        if not self._validate_task(task):
            raise ValueError("Invalid task format")
            
        task_type = task.get('type')
        url = task.get('url')
        depth = task.get('depth', 2)
        context = task.get('context', {})
        
        # Process shared context from other agents
        shared_results = context.get('shared_results', {})
        previous_results = context.get('previous_results', {})
        
        # Initialize result structure
        results = {
            'success': False,
            'data': {},
            'errors': [],
            'feedback': {
                'state_updates': {},
                'memory_updates': []
            }
        }
        
        try:
            if task_type == 'crawl':
                # Use SEO insights to guide crawling
                seo_data = shared_results.get('technical_seo', {})
                keyword_data = shared_results.get('keyword_research', {})
                
                results['data'] = await self._crawl_with_context(
                    url,
                    depth,
                    seo_data,
                    keyword_data
                )
                
            elif task_type == 'structure':
                # Use previous crawl data if available
                crawl_data = shared_results.get('crawl', {})
                results['data'] = await self._analyze_structure_with_context(
                    url,
                    crawl_data
                )
                
            elif task_type == 'content':
                # Use keyword and SEO insights for content extraction
                keyword_data = shared_results.get('keyword_research', {})
                seo_data = shared_results.get('technical_seo', {})
                
                results['data'] = await self._extract_content_with_context(
                    url,
                    keyword_data,
                    seo_data
                )
                
            elif task_type == 'links':
                results['data'] = await self._check_links_with_context(
                    url,
                    shared_results
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
            results['success'] = True
            
            # Share insights with other agents
            self._share_insights(results['data'], task_type)
            
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            results['errors'].append(str(e))
            
        return results
        
    async def _crawl_with_context(
        self,
        url: str,
        depth: int,
        seo_data: Dict[str, Any],
        keyword_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crawl site with enhanced context awareness."""
        # Try cache first
        cache_key = f"{url}_{depth}"
        if cached_data := self.cache.load(cache_key, 'crawl'):
            self.logger.info(f"Using cached crawl data for {url}")
            return cached_data
            
        # Get base crawl results
        crawl_results = await self._crawl_site(url)
        
        # Enhance crawl with SEO insights
        if seo_data:
            crawl_results['seo_analysis'] = {
                'technical_issues': self.crawler.analyze_technical_issues(crawl_results, seo_data),
                'optimization_opportunities': self.crawler.find_optimization_opportunities(
                    crawl_results, seo_data
                )
            }
            
        # Enhance with keyword insights
        if keyword_data:
            crawl_results['keyword_analysis'] = {
                'keyword_presence': self.crawler.analyze_keyword_presence(
                    crawl_results, keyword_data
                ),
                'content_gaps': self.crawler.identify_content_gaps(
                    crawl_results, keyword_data
                )
            }
            
        # Cache results
        self.cache.save(cache_key, 'crawl', crawl_results)
        
        return crawl_results
        
    async def _analyze_structure_with_context(
        self,
        url: str,
        crawl_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze site structure with existing crawl data."""
        if crawl_data:
            # Use existing crawl data for analysis
            structure_analysis = self.crawler.analyze_site_structure(crawl_data)
        else:
            # Perform new structure analysis
            structure_analysis = await self._analyze_structure(url)
            
        return {
            'site_structure': structure_analysis,
            'navigation_analysis': self.crawler.analyze_navigation(structure_analysis),
            'depth_distribution': self.crawler.analyze_depth_distribution(structure_analysis)
        }
        
    async def _extract_content_with_context(
        self,
        url: str,
        keyword_data: Dict[str, Any],
        seo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and analyze content with keyword and SEO context."""
        # Get base content extraction
        content_data = await self._extract_content(url)
        
        # Enhance with keyword insights
        if keyword_data:
            content_data['keyword_analysis'] = self.crawler.analyze_content_keywords(
                content_data['content'],
                keyword_data
            )
            
        # Enhance with SEO insights
        if seo_data:
            content_data['seo_analysis'] = self.crawler.analyze_content_seo(
                content_data['content'],
                seo_data
            )
            
        return content_data
        
    async def _check_links_with_context(
        self,
        url: str,
        shared_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check links with comprehensive context."""
        # Get base link analysis
        link_analysis = await self._check_links(url)
        
        # Enhance with insights from other agents
        seo_data = shared_results.get('technical_seo', {})
        if seo_data:
            link_analysis['seo_impact'] = self.crawler.analyze_link_seo_impact(
                link_analysis,
                seo_data
            )
            
        return link_analysis
        
    def _share_insights(self, results: Dict[str, Any], task_type: str) -> None:
        """Share crawler insights with collaborating agents."""
        insight = {
            'type': 'crawler_insight',
            'task_type': task_type,
            'data': results,
            'priority': 'high',
            'related_tasks': ['technical_seo', 'content_analysis', 'keyword_research']
        }
        
        self.remember(insight)
        # This will trigger _broadcast_observation to collaborators
        
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate crawler task format."""
        required_fields = ['type', 'url']
        return all(field in task for field in required_fields)

    async def _crawl_site(self, url: str) -> Dict[str, Any]:
        """Crawl a website and extract content and structure."""
        try:
            await self.crawler.crawl_site()
            return {
                'success': True,
                'data': self.crawler.analyze_content()
            }
        except Exception as e:
            self.logger.error(f"Error crawling site: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _analyze_structure(self, url: str) -> Dict[str, Any]:
        """Analyze site structure and navigation."""
        try:
            analysis = await self.crawler._analyze_structure_with_context(url, {})
            return {
                'success': True,
                'data': analysis
            }
        except Exception as e:
            self.logger.error(f"Error analyzing structure: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _extract_content(self, url: str) -> Dict[str, Any]:
        """Extract and analyze page content."""
        try:
            content_data = await self.crawler._extract_content_with_context(url, {}, {})
            return {
                'success': True,
                'data': content_data
            }
        except Exception as e:
            self.logger.error(f"Error extracting content: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _check_links(self, url: str) -> Dict[str, Any]:
        """Check internal and external links."""
        try:
            link_analysis = await self.crawler._check_links_with_context(url, {})
            return {
                'success': True,
                'data': link_analysis
            }
        except Exception as e:
            self.logger.error(f"Error checking links: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            } 