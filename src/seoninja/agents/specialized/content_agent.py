"""Content Generation Agent implementation."""
from typing import Dict, Any, List
from langchain.agents import Tool
from ...agents.base.agent import BaseAgent
from ...core.content_generator import ContentGenerator
from ...utils.cache import Cache

class ContentAgent(BaseAgent):
    """Agent specialized in content generation and analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Content agent with its specific tools."""
        super().__init__("content")
        self.config = config or {}
        
        # Initialize without config first
        self.content_generator = ContentGenerator(
            llm=self.config.get('openai_key')  # ContentGenerator expects llm parameter, not api_key
        )
        
        self.cache = Cache('content')
        self._setup_tools()
        
    def _setup_tools(self) -> List[Tool]:
        """Set up content generation specific tools."""
        return [
            Tool(
                name="generate_content",
                func=self._generate_content,
                description="Generates SEO-optimized content"
            ),
            Tool(
                name="analyze_content",
                func=self._analyze_content,
                description="Analyzes content quality and SEO metrics"
            ),
            Tool(
                name="optimize_content",
                func=self._optimize_content,
                description="Optimizes content for better SEO performance"
            )
        ]
        
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a content task with enhanced context awareness."""
        if not self._validate_task(task):
            raise ValueError("Invalid task format")
            
        task_type = task.get('type')
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
            if task_type == 'generate':
                # Use keyword insights for better content generation
                keyword_data = shared_results.get('keyword_research', {})
                seo_context = shared_results.get('technical_seo', {})
                
                results['data'] = await self._generate_with_context(
                    task.get('keyword_data', {}),
                    keyword_data,
                    seo_context
                )
                
            elif task_type == 'analyze':
                # Use SEO insights for better content analysis
                seo_data = shared_results.get('technical_seo', {})
                keyword_context = shared_results.get('keyword_research', {})
                
                results['data'] = await self._analyze_with_context(
                    content,
                    task.get('keywords', []),
                    seo_data,
                    keyword_context
                )
                
            elif task_type == 'optimize':
                # Combine insights from multiple agents
                results['data'] = await self._optimize_with_context(
                    content,
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
        
    async def _generate_with_context(
        self,
        keyword_data: Dict[str, Any],
        keyword_insights: Dict[str, Any],
        seo_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate content using insights from other agents."""
        # Extract relevant insights
        search_intent = keyword_insights.get('intent_analysis', {})
        seo_requirements = seo_context.get('requirements', {})
        
        # Generate enhanced content
        content = await self.content_generator.generate(
            keyword_data,
            search_intent=search_intent,
            seo_requirements=seo_requirements
        )
        
        # Add generation metadata
        return {
            'content': content,
            'metadata': {
                'keyword_data': keyword_data,
                'search_intent': search_intent,
                'seo_requirements': seo_requirements
            }
        }
        
    async def _analyze_with_context(
        self,
        content: str,
        keywords: List[str],
        seo_data: Dict[str, Any],
        keyword_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze content with context from other agents."""
        # Get base analysis
        base_analysis = await self._analyze_content(content, keywords)
        
        # Enhance analysis with SEO insights
        if seo_data:
            base_analysis['seo_alignment'] = self.content_generator.analyze_seo_alignment(
                content, seo_data
            )
            
        # Enhance with keyword insights
        if keyword_context:
            base_analysis['keyword_effectiveness'] = self.content_generator.analyze_keyword_usage(
                content, keyword_context
            )
            
        return base_analysis
        
    async def _optimize_with_context(
        self,
        content: str,
        shared_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize content using all available context."""
        # Gather insights from all agents
        seo_insights = shared_results.get('technical_seo', {})
        keyword_insights = shared_results.get('keyword_research', {})
        content_analysis = shared_results.get('analyze_content', {})
        
        # Perform comprehensive optimization
        optimized_content = await self.content_generator.optimize(
            content,
            seo_requirements=seo_insights.get('requirements', {}),
            keyword_data=keyword_insights,
            content_analysis=content_analysis
        )
        
        return {
            'original_content': content,
            'optimized_content': optimized_content,
            'optimization_details': {
                'seo_improvements': self.content_generator.get_optimization_details(),
                'keyword_improvements': self.content_generator.get_keyword_optimization_details(),
                'readability_improvements': self.content_generator.get_readability_improvements()
            }
        }
        
    def _share_insights(self, results: Dict[str, Any], task_type: str) -> None:
        """Share content insights with collaborating agents."""
        insight = {
            'type': 'content_insight',
            'task_type': task_type,
            'data': results,
            'priority': 'high',
            'related_tasks': ['seo_optimization', 'keyword_research']
        }
        
        self.remember(insight)
        # This will trigger _broadcast_observation to collaborators
        
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate content task format."""
        required_fields = ['type']
        if task['type'] in ['analyze', 'optimize']:
            required_fields.append('content')
        elif task['type'] == 'generate':
            required_fields.append('keyword_data')
        return all(field in task for field in required_fields)

    async def _generate_content(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SEO-optimized content."""
        try:
            keyword = task.get('keyword', '')
            intent = task.get('intent', 'informational')
            
            content = await self.content_generator.generate(
                keyword=keyword,
                search_intent=intent
            )
            
            return {
                'success': True,
                'data': content
            }
        except Exception as e:
            self.logger.error(f"Error generating content: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _analyze_content(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content quality and SEO metrics."""
        try:
            content = task.get('content', '')
            keywords = task.get('keywords', [])
            
            analysis = await self.content_generator.analyze_content(
                content=content,
                keywords=keywords
            )
            
            return {
                'success': True,
                'data': analysis
            }
        except Exception as e:
            self.logger.error(f"Error analyzing content: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _optimize_content(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content for better SEO performance."""
        try:
            content = task.get('content', '')
            keywords = task.get('keywords', [])
            seo_data = task.get('seo_data', {})
            
            optimized = await self.content_generator.optimize(
                content=content,
                keywords=keywords,
                seo_requirements=seo_data
            )
            
            return {
                'success': True,
                'data': optimized
            }
        except Exception as e:
            self.logger.error(f"Error optimizing content: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            } 