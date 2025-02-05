"""SEO Optimizer Agent implementation."""
from typing import Dict, Any, List
from langchain.agents import Tool
from ..base.agent import BaseAgent
from ...core.seo.optimizer import SEOOptimizer
from ...utils.cache import Cache

class SEOOptimizerAgent(BaseAgent):
    """Agent specialized in SEO optimization and technical analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize SEO Optimizer agent with its specific tools."""
        super().__init__("seo_optimizer")
        self.config = config or {}
        self.optimizer = SEOOptimizer(config=self.config)
        self.cache = Cache('seo_optimizer')
        self._setup_tools()
        
    def _setup_tools(self) -> List[Tool]:
        """Set up SEO optimization specific tools."""
        return [
            Tool(
                name="analyze_technical_seo",
                func=self._analyze_technical_seo,
                description="Analyzes technical SEO aspects of a webpage"
            ),
            Tool(
                name="optimize_meta_tags",
                func=self._optimize_meta_tags,
                description="Optimizes meta tags and structured data"
            ),
            Tool(
                name="analyze_performance",
                func=self._analyze_performance,
                description="Analyzes page performance metrics"
            ),
            Tool(
                name="generate_recommendations",
                func=self._generate_recommendations,
                description="Generates SEO improvement recommendations"
            )
        ]
        
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an SEO optimization task with enhanced context awareness."""
        if not self._validate_task(task):
            raise ValueError("Invalid task format")
            
        task_type = task.get('type')
        url = task.get('url')
        content = task.get('content')
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
            if task_type == 'technical_seo':
                # Use crawler insights for technical analysis
                crawler_data = shared_results.get('crawl', {})
                content_data = shared_results.get('analyze_content', {})
                
                results['data'] = await self._analyze_technical_with_context(
                    url or content,
                    crawler_data,
                    content_data
                )
                
            elif task_type == 'meta_optimization':
                # Use keyword and content insights for meta optimization
                keyword_data = shared_results.get('keyword_research', {})
                content_analysis = shared_results.get('analyze_content', {})
                
                results['data'] = await self._optimize_meta_with_context(
                    url or content,
                    keyword_data,
                    content_analysis
                )
                
            elif task_type == 'performance':
                results['data'] = await self._analyze_performance_with_context(
                    url,
                    shared_results
                )
                
            elif task_type == 'recommendations':
                # Combine all available insights for recommendations
                results['data'] = await self._generate_recommendations_with_context(
                    url or content,
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
        
    async def _analyze_technical_with_context(
        self,
        url: str,
        crawler_data: Dict[str, Any],
        content_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze technical SEO with context from other analyses."""
        try:
            # Get base technical analysis
            base_analysis = await self._analyze_technical_seo(url)
            
            # Enhance with crawler insights
            if crawler_data:
                base_analysis['crawl_issues'] = crawler_data.get('issues', [])
                base_analysis['site_structure'] = crawler_data.get('structure', {})
                
            # Add content insights
            if content_data:
                base_analysis['content_issues'] = content_data.get('issues', [])
                
            return base_analysis
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis with context: {str(e)}")
            return {'error': str(e)}
        
    async def _optimize_meta_with_context(
        self,
        content: str,
        keyword_data: Dict[str, Any],
        content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize meta tags with keyword and content context."""
        try:
            results = {
                'meta_tags': {},
                'recommendations': [],
                'optimization_score': 0.0
            }
            
            # Use optimizer for meta optimization
            meta_optimization = await self.optimizer.optimize_meta_tags(
                content,
                keyword_data,
                content_analysis
            )
            
            results.update(meta_optimization)
            return results
            
        except Exception as e:
            self.logger.error(f"Error in meta optimization: {str(e)}")
            return {'error': str(e)}
        
    async def _analyze_performance_with_context(
        self,
        url: str,
        shared_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance with shared context."""
        try:
            results = {
                'performance_metrics': {},
                'optimization_suggestions': [],
                'critical_issues': []
            }
            
            # Use optimizer for performance analysis
            performance_data = await self.optimizer.analyze_performance(url)
            results.update(performance_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in performance analysis: {str(e)}")
            return {'error': str(e)}
        
    async def _generate_recommendations_with_context(
        self,
        content: str,
        shared_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate SEO recommendations with full context."""
        try:
            recommendations = {
                'technical': [],
                'content': [],
                'meta': [],
                'performance': [],
                'priority_actions': []
            }
            
            # Combine insights from all analyses
            technical_data = shared_results.get('technical_seo', {})
            content_data = shared_results.get('analyze_content', {})
            keyword_data = shared_results.get('keyword_research', {})
            
            # Generate comprehensive recommendations
            recommendations.update(
                await self.optimizer.generate_recommendations(
                    content,
                    technical_data,
                    content_data,
                    keyword_data
                )
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {'error': str(e)}
        
    def _share_insights(self, results: Dict[str, Any], task_type: str) -> None:
        """Share SEO insights with collaborating agents."""
        insight = {
            'type': 'seo_insight',
            'task_type': task_type,
            'data': results,
            'priority': 'high',
            'related_tasks': ['content_generation', 'keyword_research']
        }
        
        self.remember(insight)
        # This will trigger _broadcast_observation to collaborators
        
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate SEO optimization task format."""
        required_fields = ['type']
        if task['type'] != 'recommendations':
            required_fields.append('url')
        return all(field in task for field in required_fields)

    async def _analyze_technical_seo(self, url: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze technical SEO aspects of a URL."""
        try:
            results = {
                'url': url,
                'technical_issues': [],
                'performance_metrics': {},
                'recommendations': []
            }
            
            # Use optimizer to analyze technical aspects
            technical_analysis = await self.optimizer.analyze_technical(url)
            
            # Process results
            results['technical_issues'] = technical_analysis.get('issues', [])
            results['performance_metrics'] = technical_analysis.get('metrics', {})
            results['recommendations'] = technical_analysis.get('recommendations', [])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in technical SEO analysis: {str(e)}")
            return {
                'url': url,
                'error': str(e),
                'technical_issues': [],
                'performance_metrics': {},
                'recommendations': []
            }

    async def _optimize_meta_tags(self, url: str) -> Dict[str, Any]:
        """Optimize meta tags for a URL."""
        try:
            results = {
                'url': url,
                'meta_tags': {},
                'recommendations': [],
                'optimization_score': 0.0
            }
            
            # Use optimizer to analyze and optimize meta tags
            meta_analysis = await self.optimizer.optimize_meta_tags(
                url,
                {},  # Empty keyword data as this is base analysis
                {}   # Empty content analysis as this is base analysis
            )
            
            # Process results
            results.update(meta_analysis)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in meta tag optimization: {str(e)}")
            return {
                'url': url,
                'error': str(e),
                'meta_tags': {},
                'recommendations': [],
                'optimization_score': 0.0
            }

    async def _analyze_performance(self, url: str) -> Dict[str, Any]:
        """Analyze performance metrics for a URL."""
        try:
            results = {
                'url': url,
                'performance_metrics': {},
                'optimization_suggestions': [],
                'critical_issues': []
            }
            
            # Use optimizer for performance analysis
            performance_data = await self.optimizer.analyze_performance(url)
            results.update(performance_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in performance analysis: {str(e)}")
            return {
                'url': url,
                'error': str(e),
                'performance_metrics': {},
                'optimization_suggestions': [],
                'critical_issues': []
            }

    async def _generate_recommendations(self, url: str) -> Dict[str, Any]:
        """Generate SEO recommendations for a URL."""
        try:
            results = {
                'url': url,
                'recommendations': {
                    'technical': [],
                    'content': [],
                    'meta': [],
                    'performance': [],
                    'priority_actions': []
                }
            }
            
            # Use optimizer to generate recommendations
            recommendations = await self.optimizer.generate_recommendations(
                url,
                {},  # Empty technical data as this is base analysis
                {},  # Empty content data as this is base analysis
                {}   # Empty keyword data as this is base analysis
            )
            
            results['recommendations'].update(recommendations)
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {
                'url': url,
                'error': str(e),
                'recommendations': {
                    'technical': [],
                    'content': [],
                    'meta': [],
                    'performance': [],
                    'priority_actions': []
                }
            } 