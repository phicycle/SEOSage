"""Keyword Research Agent implementation."""
from typing import Dict, Any, List
from langchain.agents import Tool
from ..base.agent import BaseAgent
from ...core.keyword_research import KeywordResearch
from ...utils.cache import Cache

class KeywordResearchAgent(BaseAgent):
    """Agent specialized in keyword research and analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Keyword Research agent with its specific tools."""
        super().__init__("keyword_research")
        self.config = config or {}
        
        # Initialize KeywordResearch without config first
        self.keyword_research = KeywordResearch()
        
        # Set config after initialization
        self.keyword_research.config = {
            'api_key': self.config.get('moz_token'),
            'api_base_url': self.config.get('api_base_url', 'https://api.keywordtool.io/v1'),
            'batch_size': self.config.get('batch_size', 50),
            'rate_limit': self.config.get('rate_limit', 1.0)
        }
        
        # Update internal variables based on new config
        self.keyword_research._api_key = self.keyword_research.config.get('api_key')
        self.keyword_research._api_base_url = self.keyword_research.config.get('api_base_url')
        self.keyword_research._batch_size = self.keyword_research.config.get('batch_size')
        self.keyword_research._rate_limit = self.keyword_research.config.get('rate_limit')
        
        self.cache = Cache('keyword_research')
        self._setup_tools()
        
    def _setup_tools(self) -> List[Tool]:
        """Set up keyword research specific tools."""
        return [
            Tool(
                name="research_keywords",
                func=self._research_keywords,
                description="Finds relevant keywords and opportunities for a domain"
            ),
            Tool(
                name="analyze_competition",
                func=self._analyze_competition,
                description="Analyzes keyword competition and difficulty"
            ),
            Tool(
                name="find_opportunities",
                func=self._find_opportunities,
                description="Identifies keyword opportunities based on metrics"
            ),
            Tool(
                name="analyze_intent",
                func=self._analyze_intent,
                description="Analyzes search intent for keywords"
            )
        ]
        
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a keyword research task with enhanced context awareness."""
        if not self._validate_task(task):
            raise ValueError("Invalid task format")
            
        task_type = task.get('type')
        domain = task.get('domain')
        keywords = task.get('keywords', [])
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
            # Adapt execution based on available context
            if task_type == 'research':
                # Check if we have relevant crawler data
                crawler_data = shared_results.get('crawl', {})
                if crawler_data:
                    self.remember({
                        'type': 'context_received',
                        'source': 'crawler',
                        'content': crawler_data,
                        'priority': 'high'
                    })
                    # Use crawler data to enhance keyword research
                    results['data'] = await self._research_with_crawler_data(domain, crawler_data)
                else:
                    results['data'] = await self._research_keywords(domain)
                    
            elif task_type == 'competition':
                # Check for content analysis context
                content_analysis = shared_results.get('analyze_content', {})
                if content_analysis:
                    # Use content analysis to refine competition analysis
                    results['data'] = await self._analyze_competition_with_content(
                        domain, keywords, content_analysis
                    )
                else:
                    results['data'] = await self._analyze_competition(domain, keywords)
                    
            elif task_type == 'opportunities':
                results['data'] = await self._find_opportunities_with_context(
                    domain, shared_results
                )
                
            elif task_type == 'intent':
                # Check for SEO context
                seo_context = shared_results.get('technical_seo', {})
                results['data'] = await self._analyze_intent_with_context(
                    keywords, seo_context
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
            results['success'] = True
            
            # Share relevant insights with other agents
            self._share_insights(results['data'], task_type)
            
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            results['errors'].append(str(e))
            
        return results
        
    async def _research_with_crawler_data(self, domain: str, crawler_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced keyword research using crawler data."""
        # Extract relevant information from crawler data
        page_content = crawler_data.get('page_content', {})
        existing_keywords = crawler_data.get('existing_keywords', [])
        
        # Get base research results
        results = await self._research_keywords(domain)
        
        # Enhance results with crawler insights
        results['crawler_enhanced'] = {
            'content_based_suggestions': self.keyword_research.suggest_from_content(page_content),
            'existing_keyword_performance': self.keyword_research.analyze_existing_keywords(existing_keywords)
        }
        
        return results
        
    async def _analyze_competition_with_content(
        self, domain: str, keywords: List[str], content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced competition analysis using content analysis data."""
        base_analysis = await self._analyze_competition(domain, keywords)
        
        # Enhance with content insights
        content_topics = content_analysis.get('main_topics', [])
        competitor_content = content_analysis.get('competitor_content', {})
        
        enhanced_analysis = {
            **base_analysis,
            'topic_competition': self.keyword_research.analyze_topic_competition(content_topics),
            'content_gap_opportunities': self.keyword_research.find_content_gaps(
                competitor_content, base_analysis
            )
        }
        
        return enhanced_analysis
        
    async def _find_opportunities_with_context(
        self, domain: str, shared_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find keyword opportunities using all available context."""
        opportunities = await self._find_opportunities(domain)
        
        # Enhance with insights from other agents
        seo_data = shared_results.get('technical_seo', {})
        content_data = shared_results.get('analyze_content', {})
        
        if seo_data or content_data:
            opportunities['contextual_opportunities'] = self.keyword_research.find_contextual_opportunities(
                domain, seo_data, content_data
            )
            
        return opportunities
        
    async def _analyze_intent_with_context(
        self, keywords: List[str], seo_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze search intent with SEO context."""
        base_intent = await self._analyze_intent(keywords)
        
        if seo_context:
            # Enhance intent analysis with SEO insights
            enhanced_intent = self.keyword_research.enhance_intent_analysis(
                base_intent, seo_context
            )
            return enhanced_intent
            
        return base_intent
        
    def _share_insights(self, results: Dict[str, Any], task_type: str) -> None:
        """Share relevant insights with collaborating agents."""
        insight = {
            'type': 'keyword_insight',
            'task_type': task_type,
            'data': results,
            'priority': 'high',
            'related_tasks': ['content_generation', 'seo_optimization']
        }
        
        self.remember(insight)
        # This will trigger _broadcast_observation to collaborators
        
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate keyword research task format."""
        required_fields = ['type']
        if task.get('type') != 'intent':
            required_fields.append('domain')
        return all(field in task for field in required_fields)
        
    async def _research_keywords(self, domain: str) -> Dict[str, Any]:
        """Research keywords for a domain."""
        # Try to load from cache first
        cached_data = self.cache.load(domain, 'research')
        if cached_data:
            self.logger.info(f"Using cached keyword research for {domain}")
            return cached_data
            
        # Get ranking opportunities
        opportunities = self.keyword_research.get_ranking_opportunities(domain)
        
        # Get keyword suggestions
        suggestions = self.keyword_research.get_keyword_suggestions(domain)
        
        # Combine results
        results = {
            'opportunities': opportunities,
            'suggestions': suggestions,
            'metrics': self.keyword_research.get_domain_metrics(domain)
        }
        
        # Cache results
        self.cache.save(domain, 'research', results)
        
        return results
        
    async def _analyze_competition(self, domain: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze keyword competition."""
        # Try to load from cache first
        cache_key = f"{domain}_{'_'.join(keywords)}"
        cached_data = self.cache.load(cache_key, 'competition')
        if cached_data:
            self.logger.info(f"Using cached competition analysis for {domain}")
            return cached_data
            
        # Get competition analysis
        analysis = self.keyword_research.analyze_competition(domain, keywords)
        
        # Cache results
        self.cache.save(cache_key, 'competition', analysis)
        
        return analysis
        
    async def _find_opportunities(self, domain: str) -> Dict[str, Any]:
        """Find keyword opportunities."""
        # Get keyword research data
        research_data = await self._research_keywords(domain)
        
        # Analyze opportunities
        return self.keyword_research.analyze_opportunities(
            research_data['opportunities'],
            research_data['metrics']
        )
        
    async def _analyze_intent(self, keywords: List[str]) -> Dict[str, Any]:
        """Analyze search intent for keywords."""
        return self.keyword_research.analyze_search_intent(keywords) 