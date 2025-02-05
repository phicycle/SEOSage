"""Orchestrator agent for coordinating specialized agents."""
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from datetime import datetime
import time
from ...agents.base.agent import BaseAgent
from ...agents.specialized.content_agent import ContentAgent
from ...agents.specialized.seo_optimizer import SEOOptimizerAgent
from ...agents.specialized.crawler_agent import CrawlerAgent
from ...agents.specialized.keyword_agent import KeywordResearchAgent
from ...utils.storage import PersistentStorage
from ...utils.security import APIKeyManager, RateLimiter, InputValidator
from ...utils.parallel import run_parallel_tasks
from ...config.settings import get_settings

class TaskDecomposer:
    """Decomposes complex tasks into subtasks with enhanced context sharing."""
    
    def __init__(self):
        self.task_dependencies = {
            'website_analysis': ['crawl', 'analyze_content', 'keyword_research'],
            'content_generation': ['keyword_research', 'generate', 'optimize'],
            'content_optimization': ['technical_seo', 'analyze', 'intent']
        }
        
    async def decompose_task(self, task: Dict[str, Any], shared_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Decompose task into subtasks with context preservation."""
        subtasks = []
        task_context = shared_context or {}
        
        if task['type'] == 'website_analysis':
            # Sequential crawl and analysis
            crawl_task = {
                'type': 'crawl',
                'url': task['url'],
                'depth': task.get('depth', 2),
                'service': 'crawler',
                'context': task_context
            }
            crawl_result = await self._execute_subtask(crawl_task)
            task_context['crawl_data'] = crawl_result.get('data', {})
            
            subtasks.extend([
                {
                    'type': 'analyze_content',
                    'url': task['url'],
                    'service': 'content',
                    'context': task_context,
                    'dependencies': ['crawl']
                },
                {
                    'type': 'keyword_research',
                    'domain': task['url'],
                    'service': 'keyword',
                    'context': task_context,
                    'dependencies': ['crawl']
                }
            ])
            
        elif task['type'] == 'content_generation':
            # Sequential keyword research and content generation
            keyword_task = {
                'type': 'keyword_research',
                'keyword': task['keyword'],
                'service': 'keyword',
                'context': task_context
            }
            keyword_result = await self._execute_subtask(keyword_task)
            task_context['keyword_data'] = keyword_result.get('data', {})
            
            subtasks.extend([
                {
                    'type': 'generate',
                    'keyword_data': task_context['keyword_data'],
                    'service': 'content',
                    'context': task_context,
                    'dependencies': ['keyword_research']
                },
                {
                    'type': 'optimize',
                    'keyword_data': task_context['keyword_data'],
                    'service': 'seo',
                    'context': task_context,
                    'dependencies': ['generate']
                }
            ])
            
        elif task['type'] == 'content_optimization':
            # Parallel optimization tasks
            subtasks = [
                {
                    'type': 'technical_seo',
                    'content': task['content'],
                    'service': 'seo',
                    'context': task_context
                },
                {
                    'type': 'analyze',
                    'content': task['content'],
                    'keywords': task.get('keywords', []),
                    'service': 'content',
                    'context': task_context
                },
                {
                    'type': 'intent',
                    'keywords': task.get('keywords', []),
                    'service': 'keyword',
                    'context': task_context
                }
            ]
            
        return subtasks
        
    async def _execute_subtask(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask and return its result."""
        # This would be implemented by the orchestrator
        pass

class BatchExecutor:
    """Handles batched execution of tasks with retries."""
    
    def __init__(self, batch_size: int = 5, max_retries: int = 3, retry_delay: float = 1.0):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    async def execute(self, tasks: List[Dict[str, Any]], executor_func) -> List[Dict[str, Any]]:
        """Execute tasks in batches with retry logic."""
        results = []
        
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            batch_results = await self._execute_batch(batch, executor_func)
            results.extend(batch_results)
            
        return results
        
    async def _execute_batch(self, batch: List[Dict[str, Any]], executor_func) -> List[Dict[str, Any]]:
        """Execute a single batch with retries."""
        for attempt in range(self.max_retries):
            try:
                return await asyncio.gather(
                    *[executor_func(task) for task in batch],
                    return_exceptions=True
                )
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))

class SEOOrchestrator(BaseAgent):
    """Orchestrates multiple specialized agents for SEO tasks with enhanced coordination."""
    
    def __init__(
        self,
        storage: Optional[PersistentStorage] = None,
        gsc_credentials: Optional[Dict[str, Any]] = None,
        moz_token: Optional[str] = None,
        target_domain: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """Initialize orchestrator with enhanced components."""
        super().__init__("orchestrator")
        
        # Initialize components
        self.storage = storage or PersistentStorage()
        self.gsc_credentials = gsc_credentials
        self.moz_token = moz_token
        self.target_domain = target_domain
        self.openai_api_key = openai_api_key
        
        # Initialize config
        self.config = {
            'gsc_credentials': self.gsc_credentials,
            'target_domain': self.target_domain,
            'moz_token': self.moz_token,
            'openai_api_key': self.openai_api_key
        }
        
        self.api_keys = APIKeyManager()
        self.rate_limiter = RateLimiter()
        self.validator = InputValidator()
        self.task_decomposer = TaskDecomposer()
        self.batch_executor = BatchExecutor()
        
        # Set up rate limits
        self._setup_rate_limits()
        
        # Initialize empty agents dict
        self.agents: Dict[str, BaseAgent] = {}
        
        # Initialize monitoring intervals
        self.monitoring_interval = 60  # seconds
        self.coordination_interval = 1  # seconds
        
        # Initialize task queue
        self.task_queue = asyncio.Queue()
        
        # Load previous state if exists
        self._load_state()
        
        # Shared results for real-time updates
        self.shared_results = {}
        
    def _setup_rate_limits(self) -> None:
        """Set up rate limits for various services."""
        self.rate_limiter.add_limit('openai', 60, 60)  # 60 requests per minute
        self.rate_limiter.add_limit('moz', 10, 60)     # 10 requests per minute
        self.rate_limiter.add_limit('gsc', 100, 60)    # 100 requests per minute
        self.rate_limiter.add_limit('crawler', 100, 60) # 100 requests per minute
        
    def _initialize_agents(self) -> None:
        """Initialize specialized agents with shared context."""
        self.agents = {
            'crawler': CrawlerAgent(config={
                'domain': self.target_domain,
                'rate_limit': 1.0,
                'max_pages': 1000,
                'max_depth': 5
            }),
            'content': ContentAgent(config={
                'openai_key': self.openai_api_key
            }),
            'keyword': KeywordResearchAgent(config={
                'moz_token': self.moz_token,
                'api_base_url': 'https://api.keywordtool.io/v1',
                'batch_size': 50,
                'rate_limit': 1.0
            }),
            'seo': SEOOptimizerAgent(config=self.config)
        }
        
    def _load_state(self) -> None:
        """Load previous state from storage."""
        if state := self.storage.get_state(self.name):
            self.state = state
            
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complex task by coordinating multiple agents with enhanced context sharing."""
        start_time = time.time()
        
        try:
            # Validate input
            if error := self._validate_task(task):
                return {'success': False, 'error': error}
                
            # Initialize shared context
            shared_context = {
                'task_id': task.get('id', str(time.time())),
                'parent_task': task['type'],
                'start_time': start_time,
                'global_parameters': task.get('parameters', {}),
                'intermediate_results': {}
            }
            
            # Decompose task
            subtasks = await self.task_decomposer.decompose_task(task, shared_context)
            
            # Execute subtasks based on task type and dependencies
            if task['type'] in ['website_analysis', 'content_optimization']:
                # Run parallel tasks with real-time updates
                results = await self._execute_parallel_tasks(subtasks)
            else:
                # Run sequential tasks with context preservation
                results = await self._execute_sequential_tasks(subtasks)
                
            # Record metrics
            execution_time = time.time() - start_time
            self.storage.save_metric(self.name, 'task_execution_time', execution_time)
            
            # Update state with comprehensive results
            self.update_state({
                'last_task': task,
                'last_results': results,
                'last_execution_time': execution_time,
                'shared_context': shared_context
            })
            
            # Save state
            self.storage.save_state(self.name, self.state)
            
            return {
                'success': True,
                'data': self._format_results(task['type'], results),
                'execution_time': execution_time,
                'context': shared_context
            }
            
        except Exception as e:
            self.log_progress(f"Task execution failed: {str(e)}", 'error')
            return {
                'success': False,
                'error': str(e),
                'partial_results': self.shared_results
            }
            
    def _validate_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Validate task has required fields."""
        if 'type' not in task:
            return "Task type is required"
            
        if task['type'] == 'website_analysis':
            if 'url' not in task:
                return "URL is required for website analysis"
            if error := self.validator.validate_url(task['url']):
                return error
                
        elif task['type'] == 'content_generation':
            if 'keyword' not in task:
                return "Keyword is required for content generation"
                
        elif task['type'] == 'content_optimization':
            if 'content' not in task:
                return "Content is required for optimization"
                
        return None
        
    async def _execute_parallel_tasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tasks in parallel with batching and error handling."""
        try:
            results = await self.batch_executor.execute(
                subtasks,
                lambda task: self._execute_with_updates(task, self.shared_results)
            )
            
            # Filter out exceptions and log errors
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task execution failed: {str(result)}")
                    processed_results.append({
                        'success': False,
                        'error': str(result)
                    })
                else:
                    processed_results.append(result)
                    
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {str(e)}")
            return [{
                'success': False,
                'error': f"Batch execution failed: {str(e)}"
            }]
            
    async def _execute_sequential_tasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tasks sequentially with enhanced error handling and state management."""
        results = []
        current_state = {}
        
        for subtask in subtasks:
            try:
                # Wait for dependencies with timeout
                await asyncio.wait_for(
                    self._wait_for_dependencies(subtask),
                    timeout=300  # 5 minutes timeout
                )
                
                # Update task with current state
                subtask['context']['current_state'] = current_state
                
                # Execute task
                result = await self._execute_with_updates(subtask, self.shared_results)
                results.append(result)
                
                # Update state if successful
                if result.get('success'):
                    current_state[subtask['type']] = result.get('data', {})
                    self.shared_results[subtask['type']] = result.get('data', {})
                else:
                    self.logger.warning(f"Task {subtask['type']} failed: {result.get('error')}")
                    if subtask.get('critical'):
                        break
                        
            except asyncio.TimeoutError:
                error_msg = f"Timeout waiting for dependencies of task {subtask['type']}"
                self.logger.error(error_msg)
                results.append({
                    'success': False,
                    'error': error_msg
                })
                if subtask.get('critical'):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error executing task {subtask['type']}: {str(e)}")
                results.append({
                    'success': False,
                    'error': str(e)
                })
                if subtask.get('critical'):
                    break
                    
        return results
        
    async def _execute_with_updates(self, subtask: Dict[str, Any], shared_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task with real-time updates."""
        agent = self.agents[subtask['service']]
        
        # Add shared results to context
        subtask['context']['shared_results'] = shared_results
        
        # Execute task
        result = await agent.execute(subtask)
        
        # Update shared results
        if result.get('success'):
            shared_results[subtask['type']] = result.get('data')
            
        return result
        
    async def _wait_for_dependencies(self, task: Dict[str, Any]) -> None:
        """Wait for task dependencies with improved handling."""
        dependencies = task.get('dependencies', [])
        start_time = time.time()
        check_interval = 0.1
        
        while dependencies:
            # Remove satisfied dependencies
            dependencies = [
                dep for dep in dependencies 
                if dep not in self.shared_results
            ]
            
            if dependencies:
                # Check timeout
                if time.time() - start_time > 300:  # 5 minutes timeout
                    raise TimeoutError(
                        f"Timeout waiting for dependencies: {', '.join(dependencies)}"
                    )
                    
                # Exponential backoff for checking interval
                check_interval = min(check_interval * 1.5, 5.0)
                await asyncio.sleep(check_interval)
                
    def _format_results(self, task_type: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format results based on task type with enhanced metadata."""
        formatted_results = {
            'task_type': task_type,
            'timestamp': datetime.now().isoformat(),
            'results': {}
        }
        
        if task_type == 'website_analysis':
            formatted_results['results'] = {
                'structure': results[0].get('data', {}),
                'content': results[1].get('data', {}),
                'keywords': results[2].get('data', {})
            }
        elif task_type == 'content_generation':
            formatted_results['results'] = {
                'keyword_research': results[0].get('data', {}),
                'content': results[1].get('data', {}),
                'optimization': results[2].get('data', {})
            }
        elif task_type == 'content_optimization':
            formatted_results['results'] = {
                'technical_seo': results[0].get('data', {}),
                'content_analysis': results[1].get('data', {}),
                'intent_analysis': results[2].get('data', {})
            }
            
        return formatted_results
        
    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        for agent in self.agents.values():
            agent.cleanup()

    async def run(self) -> None:
        """Start the orchestrator and initialize all agents."""
        try:
            self.logger.info("Starting SEO Orchestrator...")
            
            # Initialize state if not exists
            if 'agents' not in self.state:
                self.state['agents'] = {}
            
            # Initialize all agents
            await self._initialize_agents()
            
            # Start monitoring and coordination
            await self._start_monitoring()
            
            self.logger.info("SEO Orchestrator running")
            
        except Exception as e:
            self.logger.error(f"Failed to start orchestrator: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator and all agents."""
        try:
            self.logger.info("Shutting down SEO Orchestrator...")
            
            # Stop all running tasks
            await self._stop_all_tasks()
            
            # Cleanup agents
            await self._cleanup_agents()
            
            # Save state if needed
            await self._save_state()
            
            self.logger.info("SEO Orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise

    async def _initialize_agents(self) -> None:
        """Initialize all specialized agents."""
        try:
            # Initialize each specialized agent
            self.agents = {
                'keyword': KeywordResearchAgent(self.config.get('keyword', {})),
                'content': ContentAgent(self.config.get('content', {})),
                'seo': SEOOptimizerAgent(self.config.get('seo', {})),
                'crawler': CrawlerAgent(self.config.get('crawler', {}))
            }
            
            # Setup agent collaboration
            for agent_name, agent in self.agents.items():
                await self._setup_agent_collaboration(agent)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {str(e)}")
            raise

    async def _start_monitoring(self) -> None:
        """Start monitoring agent activities and task execution."""
        try:
            self.monitoring_task = asyncio.create_task(self._monitor_agents())
            self.coordination_task = asyncio.create_task(self._coordinate_tasks())
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {str(e)}")
            raise

    async def _stop_all_tasks(self) -> None:
        """Stop all running tasks gracefully."""
        try:
            # Cancel monitoring and coordination tasks
            if hasattr(self, 'monitoring_task'):
                self.monitoring_task.cancel()
                
            if hasattr(self, 'coordination_task'):
                self.coordination_task.cancel()
                
            # Wait for tasks to complete
            pending = [task for task in asyncio.all_tasks() 
                      if task is not asyncio.current_task()]
            await asyncio.gather(*pending, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error stopping tasks: {str(e)}")
            raise

    async def _cleanup_agents(self) -> None:
        """Cleanup and shutdown all agents."""
        try:
            for agent_name, agent in self.agents.items():
                if agent is not None:  # Add check for None
                    try:
                        await agent.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up {agent_name}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error during agent cleanup: {str(e)}")
            raise

    async def _save_state(self) -> None:
        """Save orchestrator and agent states."""
        try:
            state = {
                'orchestrator': self.state,
                'agents': {
                    name: agent.get_state() 
                    for name, agent in self.agents.items()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            await self.storage.save('orchestrator_state', state)
            self.logger.info("Orchestrator state saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")
            raise

    async def _monitor_agents(self) -> None:
        """Monitor agent health and performance."""
        while True:
            try:
                for agent_name, agent in self.agents.items():
                    health = await agent.get_health()
                    if not health['healthy']:
                        self.logger.warning(
                            f"Agent {agent_name} health check failed: {health['status']}"
                        )
                        await self._handle_agent_failure(agent_name, agent)
                        
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in agent monitoring: {str(e)}")
                await asyncio.sleep(self.monitoring_interval)

    async def _coordinate_tasks(self) -> None:
        """Coordinate task execution between agents."""
        while True:
            try:
                # Process task queue
                while not self.task_queue.empty():
                    task = await self.task_queue.get()
                    await self._process_task(task)
                    
                await asyncio.sleep(self.coordination_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task coordination: {str(e)}")
                await asyncio.sleep(self.coordination_interval)

    async def _setup_agent_collaboration(self, agent: BaseAgent) -> None:
        """Setup collaboration between agents."""
        try:
            # Register for agent observations
            agent.add_observer(self)
            
            # Share relevant context
            await agent.update_context({
                'shared_storage': self.storage,
                'global_config': self.config,
                'orchestrator_name': self.name
            })
            
            # Setup communication channels
            await self._setup_communication_channels(agent)
            
        except Exception as e:
            self.logger.error(f"Failed to setup collaboration for {agent.name}: {str(e)}")
            raise

    async def _setup_communication_channels(self, agent: BaseAgent) -> None:
        """Setup communication channels between agents."""
        try:
            # Register message handlers
            agent.register_handler('task_complete', self._handle_task_complete)
            agent.register_handler('task_failed', self._handle_task_failed)
            agent.register_handler('state_update', self._handle_state_update)
            
            # Setup agent-specific channels
            if isinstance(agent, KeywordResearchAgent):
                agent.register_handler('keyword_insights', self._handle_keyword_insights)
            elif isinstance(agent, ContentAgent):
                agent.register_handler('content_insights', self._handle_content_insights)
            elif isinstance(agent, SEOOptimizerAgent):
                agent.register_handler('seo_insights', self._handle_seo_insights)
            elif isinstance(agent, CrawlerAgent):
                agent.register_handler('crawl_insights', self._handle_crawl_insights)
            
        except Exception as e:
            self.logger.error(f"Failed to setup channels for {agent.name}: {str(e)}")
            raise

    async def _handle_task_complete(self, data: Dict[str, Any]) -> None:
        """Handle task completion."""
        try:
            task_id = data.get('task_id')
            results = data.get('results', {})
            self.shared_results[task_id] = results
            await self._notify_dependent_tasks(task_id)
        except Exception as e:
            self.logger.error(f"Error handling task completion: {str(e)}")

    async def _handle_task_failed(self, data: Dict[str, Any]) -> None:
        """Handle task failure."""
        try:
            task_id = data.get('task_id')
            error = data.get('error')
            self.logger.error(f"Task {task_id} failed: {error}")
            await self._handle_task_failure(task_id, error)
        except Exception as e:
            self.logger.error(f"Error handling task failure: {str(e)}")

    async def _handle_state_update(self, data: Dict[str, Any]) -> None:
        """Handle agent state updates."""
        try:
            agent_name = data.get('agent')
            state = data.get('state', {})
            self.state['agents'][agent_name] = state
        except Exception as e:
            self.logger.error(f"Error handling state update: {str(e)}")

    async def _handle_keyword_insights(self, data: Dict[str, Any]) -> None:
        """Handle keyword research insights."""
        self.shared_results.setdefault('keyword_insights', []).append(data)

    async def _handle_content_insights(self, data: Dict[str, Any]) -> None:
        """Handle content analysis insights."""
        self.shared_results.setdefault('content_insights', []).append(data)

    async def _handle_seo_insights(self, data: Dict[str, Any]) -> None:
        """Handle SEO optimization insights."""
        self.shared_results.setdefault('seo_insights', []).append(data)

    async def _handle_crawl_insights(self, data: Dict[str, Any]) -> None:
        """Handle crawler insights."""
        self.shared_results.setdefault('crawl_insights', []).append(data)

    async def handle_event(self, agent_name: str, event: Dict[str, Any]) -> None:
        """Handle events from observed agents."""
        try:
            event_type = event.get('type')
            if event_type in self.handlers:
                await self.handlers[event_type](event)
            else:
                self.logger.warning(f"No handler for event type: {event_type} from {agent_name}")
        except Exception as e:
            self.logger.error(f"Error handling event from {agent_name}: {str(e)}")

    async def _notify_dependent_tasks(self, completed_task_id: str) -> None:
        """Notify tasks that depend on a completed task."""
        try:
            # Find tasks that depend on the completed task
            dependent_tasks = [
                task for task in self.task_queue._queue
                if completed_task_id in task.get('dependencies', [])
            ]
            
            # Update their context with the completed task's results
            for task in dependent_tasks:
                task['context']['completed_dependencies'] = task.get(
                    'context', {}
                ).get('completed_dependencies', []) + [completed_task_id]
                
        except Exception as e:
            self.logger.error(f"Error notifying dependent tasks: {str(e)}")

    async def _handle_agent_failure(self, agent_name: str, agent: BaseAgent) -> None:
        """Handle agent failure."""
        try:
            self.logger.warning(f"Attempting to recover agent: {agent_name}")
            
            # Try to reinitialize the agent
            await self._reinitialize_agent(agent_name)
            
            # Update state
            self.state['agents'][agent_name] = {
                'status': 'recovered',
                'last_failure': datetime.now().isoformat(),
                'recovery_attempts': self.state.get('agents', {}).get(
                    agent_name, {}
                ).get('recovery_attempts', 0) + 1
            }
            
        except Exception as e:
            self.logger.error(f"Failed to recover agent {agent_name}: {str(e)}")

    async def _reinitialize_agent(self, agent_name: str) -> None:
        """Reinitialize a failed agent."""
        try:
            # Create new instance
            agent_config = self.config.get(agent_name, {})
            if agent_name == 'keyword':
                self.agents[agent_name] = KeywordResearchAgent(agent_config)
            elif agent_name == 'content':
                self.agents[agent_name] = ContentAgent(agent_config)
            elif agent_name == 'seo':
                self.agents[agent_name] = SEOOptimizerAgent(agent_config)
            elif agent_name == 'crawler':
                self.agents[agent_name] = CrawlerAgent(agent_config)
            
            # Setup collaboration
            await self._setup_agent_collaboration(self.agents[agent_name])
            
        except Exception as e:
            self.logger.error(f"Failed to reinitialize agent {agent_name}: {str(e)}")
            raise

    async def _process_task(self, task: Dict[str, Any]) -> None:
        """Process a task from the queue."""
        try:
            # Get target agent
            agent_name = task.get('service')
            if not agent_name or agent_name not in self.agents:
                raise ValueError(f"Invalid agent specified: {agent_name}")
            
            agent = self.agents[agent_name]
            
            # Execute task
            result = await agent.execute(task)
            
            # Handle result
            if result.get('success'):
                await self._handle_task_complete({
                    'task_id': task.get('id'),
                    'results': result.get('data', {})
                })
            else:
                await self._handle_task_failed({
                    'task_id': task.get('id'),
                    'error': result.get('error', 'Unknown error')
                })
            
        except Exception as e:
            self.logger.error(f"Error processing task: {str(e)}")
            await self._handle_task_failed({
                'task_id': task.get('id'),
                'error': str(e)
            })

    async def _handle_task_failure(self, task_id: str, error: str) -> None:
        """Handle task failure."""
        try:
            # Update task status
            self.state['tasks'] = self.state.get('tasks', {})
            self.state['tasks'][task_id] = {
                'status': 'failed',
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
            
            # Notify dependent tasks
            await self._notify_task_failure(task_id, error)
            
        except Exception as e:
            self.logger.error(f"Error handling task failure: {str(e)}")

    async def _notify_task_failure(self, failed_task_id: str, error: str) -> None:
        """Notify dependent tasks of failure."""
        try:
            # Find dependent tasks
            dependent_tasks = [
                task for task in self.task_queue._queue
                if failed_task_id in task.get('dependencies', [])
            ]
            
            # Update their context
            for task in dependent_tasks:
                task['context']['failed_dependencies'] = task.get(
                    'context', {}
                ).get('failed_dependencies', []) + [{
                    'task_id': failed_task_id,
                    'error': error
                }]
            
        except Exception as e:
            self.logger.error(f"Error notifying task failure: {str(e)}") 