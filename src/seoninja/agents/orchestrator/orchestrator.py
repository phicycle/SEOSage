"""Orchestrator agent for coordinating specialized agents."""
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
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
import psutil

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

    async def get_communication_flows(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        """Get inter-agent communication flows."""
        try:
            flows = []
            for agent_name, agent in self.agents.items():
                agent_flows = agent.get_communication_history()
                filtered_flows = self._filter_by_timerange(agent_flows, start_time, end_time)
                flows.extend(filtered_flows)

            return {
                'flows': flows,
                'metrics': {
                    'total_messages': len(flows),
                    'messages_per_agent': self._count_messages_per_agent(flows),
                    'communication_patterns': self._analyze_communication_patterns(flows)
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting communication flows: {str(e)}")
            return {'error': str(e)}

    def get_realtime_communication(self) -> Dict[str, Any]:
        """Get real-time agent interaction data."""
        try:
            current_interactions = {
                agent_name: {
                    'active_communications': agent.get_active_communications(),
                    'pending_messages': agent.get_pending_messages(),
                    'last_interaction': agent.get_last_interaction()
                }
                for agent_name, agent in self.agents.items()
            }

            return {
                'timestamp': datetime.now().isoformat(),
                'interactions': current_interactions,
                'active_channels': self._get_active_channels()
            }
        except Exception as e:
            self.logger.error(f"Error getting realtime communication: {str(e)}")
            return {'error': str(e)}

    async def get_communication_analytics(self, timeframe: str = '24h') -> Dict[str, Any]:
        """Get communication patterns analytics."""
        try:
            # Convert timeframe to timedelta
            time_range = self._parse_timeframe(timeframe)
            start_time = datetime.now() - time_range

            # Get communication data
            flows = await self.get_communication_flows(start_time.isoformat())
            
            return {
                'patterns': self._analyze_communication_patterns(flows.get('flows', [])),
                'metrics': {
                    'message_frequency': self._calculate_message_frequency(flows.get('flows', [])),
                    'response_times': self._calculate_response_times(flows.get('flows', [])),
                    'channel_usage': self._analyze_channel_usage(flows.get('flows', []))
                },
                'insights': self._generate_communication_insights(flows.get('flows', []))
            }
        except Exception as e:
            self.logger.error(f"Error getting communication analytics: {str(e)}")
            return {'error': str(e)}

    async def get_data_flow_metrics(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        """Get data flow metrics between agents."""
        try:
            flows = await self.get_communication_flows(start_time, end_time)
            
            return {
                'data_volume': self._calculate_data_volume(flows.get('flows', [])),
                'flow_patterns': self._analyze_flow_patterns(flows.get('flows', [])),
                'bottlenecks': self._identify_flow_bottlenecks(flows.get('flows', [])),
                'optimization_suggestions': self._generate_flow_optimization_suggestions(flows.get('flows', []))
            }
        except Exception as e:
            self.logger.error(f"Error getting data flow metrics: {str(e)}")
            return {'error': str(e)}

    # Process Timeline Methods
    async def get_process_timeline(self, process_id: str, include_details: bool = False) -> Dict[str, Any]:
        """Get detailed process stage tracking data."""
        try:
            process = self.state.get('processes', {}).get(process_id)
            if not process:
                return {'error': 'Process not found'}

            timeline = {
                'process_id': process_id,
                'stages': self._get_process_stages(process_id),
                'current_stage': self._get_current_stage(process_id),
                'duration': self._calculate_process_duration(process_id),
                'status': self._get_process_status(process_id)
            }

            if include_details:
                timeline.update({
                    'stage_details': self._get_stage_details(process_id),
                    'dependencies': self._get_process_dependencies(process_id),
                    'resources': self._get_process_resources(process_id)
                })

            return timeline
        except Exception as e:
            self.logger.error(f"Error getting process timeline: {str(e)}")
            return {'error': str(e)}

    async def get_decision_points(self, process_id: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        """Get decision point logging data."""
        try:
            decisions = self._get_process_decisions(process_id)
            filtered_decisions = self._filter_by_timerange(decisions, start_time, end_time)

            return {
                'process_id': process_id,
                'decisions': filtered_decisions,
                'metrics': {
                    'total_decisions': len(filtered_decisions),
                    'decision_types': self._analyze_decision_types(filtered_decisions),
                    'average_decision_time': self._calculate_average_decision_time(filtered_decisions)
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting decision points: {str(e)}")
            return {'error': str(e)}

    async def get_stage_transitions(self, process_id: str) -> Dict[str, Any]:
        """Get stage transition data."""
        try:
            transitions = self._get_stage_transitions(process_id)
            return {
                'process_id': process_id,
                'transitions': transitions,
                'metrics': {
                    'total_transitions': len(transitions),
                    'average_stage_duration': self._calculate_average_stage_duration(transitions),
                    'bottleneck_stages': self._identify_bottleneck_stages(transitions)
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting stage transitions: {str(e)}")
            return {'error': str(e)}

    async def get_time_analysis(self, timeframe: str, process_type: Optional[str] = None) -> Dict[str, Any]:
        """Get time-based process analysis."""
        try:
            time_range = self._parse_timeframe(timeframe)
            processes = self._get_processes_in_timeframe(time_range, process_type)

            return {
                'timeframe': timeframe,
                'process_type': process_type,
                'metrics': {
                    'average_duration': self._calculate_average_process_duration(processes),
                    'completion_rate': self._calculate_completion_rate(processes),
                    'stage_distribution': self._analyze_stage_distribution(processes)
                },
                'trends': self._analyze_time_trends(processes),
                'optimization_suggestions': self._generate_time_optimization_suggestions(processes)
            }
        except Exception as e:
            self.logger.error(f"Error getting time analysis: {str(e)}")
            return {'error': str(e)}

    # Resource Monitoring Methods
    async def get_resource_allocation(self, resource_type: Optional[str] = None, timeframe: str = '1h') -> Dict[str, Any]:
        """Get detailed resource allocation tracking."""
        try:
            time_range = self._parse_timeframe(timeframe)
            allocations = self._get_resource_allocations(time_range, resource_type)

            return {
                'resource_type': resource_type,
                'timeframe': timeframe,
                'allocations': allocations,
                'metrics': {
                    'utilization': self._calculate_resource_utilization(allocations),
                    'efficiency': self._calculate_resource_efficiency(allocations),
                    'availability': self._calculate_resource_availability(allocations)
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting resource allocation: {str(e)}")
            return {'error': str(e)}

    async def get_usage_patterns(self, resource_type: Optional[str], start_time: Optional[str], end_time: Optional[str]) -> Dict[str, Any]:
        """Get resource usage patterns."""
        try:
            usage_data = self._get_resource_usage(resource_type, start_time, end_time)
            
            return {
                'resource_type': resource_type,
                'patterns': self._analyze_usage_patterns(usage_data),
                'trends': self._analyze_usage_trends(usage_data),
                'predictions': self._predict_future_usage(usage_data)
            }
        except Exception as e:
            self.logger.error(f"Error getting usage patterns: {str(e)}")
            return {'error': str(e)}

    async def get_resource_bottlenecks(self, analysis_period: str = '24h', threshold: float = 0.8) -> Dict[str, Any]:
        """Get resource bottleneck identification."""
        try:
            time_range = self._parse_timeframe(analysis_period)
            usage_data = self._get_resource_usage(None, (datetime.now() - time_range).isoformat())
            
            bottlenecks = self._identify_bottlenecks(usage_data, threshold)
            return {
                'bottlenecks': bottlenecks,
                'impact_analysis': self._analyze_bottleneck_impact(bottlenecks),
                'recommendations': self._generate_bottleneck_recommendations(bottlenecks)
            }
        except Exception as e:
            self.logger.error(f"Error getting resource bottlenecks: {str(e)}")
            return {'error': str(e)}

    async def get_system_capacity(self, include_predictions: bool = False) -> Dict[str, Any]:
        """Get system capacity metrics."""
        try:
            current_capacity = self._get_current_capacity()
            metrics = {
                'current_capacity': current_capacity,
                'utilization': self._calculate_system_utilization(),
                'headroom': self._calculate_system_headroom(),
                'limits': self._get_system_limits()
            }

            if include_predictions:
                metrics['predictions'] = {
                    'future_capacity': self._predict_future_capacity(),
                    'growth_trends': self._analyze_capacity_trends(),
                    'scaling_recommendations': self._generate_scaling_recommendations()
                }

            return metrics
        except Exception as e:
            self.logger.error(f"Error getting system capacity: {str(e)}")
            return {'error': str(e)}

    def get_realtime_resource_metrics(self) -> Dict[str, Any]:
        """Get real-time resource metrics."""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': self._get_cpu_usage(),
                'memory_usage': self._get_memory_usage(),
                'network_usage': self._get_network_usage(),
                'agent_resources': self._get_agent_resource_usage(),
                'queue_metrics': self._get_queue_metrics()
            }
        except Exception as e:
            self.logger.error(f"Error getting realtime resource metrics: {str(e)}")
            return {'error': str(e)}

    # Helper Methods
    def _filter_by_timerange(self, data: List[Dict[str, Any]], start_time: Optional[str], end_time: Optional[str]) -> List[Dict[str, Any]]:
        """Filter data by time range."""
        if not start_time and not end_time:
            return data

        filtered_data = data
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            filtered_data = [d for d in filtered_data if datetime.fromisoformat(d['timestamp']) >= start_dt]
        if end_time:
            end_dt = datetime.fromisoformat(end_time)
            filtered_data = [d for d in filtered_data if datetime.fromisoformat(d['timestamp']) <= end_dt]

        return filtered_data

    def _parse_timeframe(self, timeframe: str) -> timedelta:
        """Parse timeframe string into timedelta."""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        elif unit == 'w':
            return timedelta(weeks=value)
        else:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

    # Private Helper Methods for Communication Analysis
    def _count_messages_per_agent(self, flows: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count messages per agent."""
        counts = {}
        for flow in flows:
            counts[flow['source']] = counts.get(flow['source'], 0) + 1
            counts[flow['target']] = counts.get(flow['target'], 0) + 1
        return counts

    def _analyze_communication_patterns(self, flows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze communication patterns between agents."""
        patterns = {
            'common_paths': self._identify_common_paths(flows),
            'frequency_matrix': self._create_frequency_matrix(flows),
            'peak_times': self._identify_peak_times(flows)
        }
        return patterns

    def _identify_common_paths(self, flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify common communication paths."""
        path_counts = {}
        for flow in flows:
            path = f"{flow['source']}->{flow['target']}"
            path_counts[path] = path_counts.get(path, 0) + 1
        return [{'path': k, 'count': v} for k, v in sorted(path_counts.items(), key=lambda x: x[1], reverse=True)]

    def _create_frequency_matrix(self, flows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Create communication frequency matrix."""
        matrix = {}
        for flow in flows:
            source = flow['source']
            target = flow['target']
            if source not in matrix:
                matrix[source] = {}
            matrix[source][target] = matrix[source].get(target, 0) + 1
        return matrix

    def _identify_peak_times(self, flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify peak communication times."""
        hour_counts = {}
        for flow in flows:
            hour = datetime.fromisoformat(flow['timestamp']).hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        return [{'hour': k, 'count': v} for k, v in sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)]

    # Private Helper Methods for Process Timeline
    def _get_process_stages(self, process_id: str) -> List[Dict[str, Any]]:
        """Get all stages of a process."""
        process = self.state.get('processes', {}).get(process_id, {})
        return process.get('stages', [])

    def _calculate_stage_duration(self, stage: Dict[str, Any]) -> float:
        """Calculate duration of a single stage."""
        start_time = datetime.fromisoformat(stage.get('start_time', datetime.now().isoformat()))
        end_time = datetime.fromisoformat(stage.get('end_time', datetime.now().isoformat()))
        return (end_time - start_time).total_seconds()

    def _get_process_decisions(self, process_id: str) -> List[Dict[str, Any]]:
        """Get decision points for a process."""
        process = self.state.get('processes', {}).get(process_id, {})
        return process.get('decisions', [])

    def _analyze_decision_types(self, decisions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze types of decisions made."""
        type_counts = {}
        for decision in decisions:
            decision_type = decision.get('type', 'unknown')
            type_counts[decision_type] = type_counts.get(decision_type, 0) + 1
        return type_counts

    def _calculate_average_decision_time(self, decisions: List[Dict[str, Any]]) -> float:
        """Calculate average time taken for decisions."""
        if not decisions:
            return 0.0
        times = []
        for decision in decisions:
            start_time = datetime.fromisoformat(decision.get('start_time', ''))
            end_time = datetime.fromisoformat(decision.get('end_time', ''))
            times.append((end_time - start_time).total_seconds())
        return sum(times) / len(times) if times else 0.0

    # Private Helper Methods for Resource Monitoring
    def _get_resource_usage(self, resource_type: Optional[str], start_time: Optional[str], end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get resource usage data."""
        usage_data = self.state.get('resource_usage', [])
        filtered = self._filter_by_timerange(usage_data, start_time, end_time)
        if resource_type:
            filtered = [u for u in filtered if u['type'] == resource_type]
        return filtered

    def _analyze_usage_patterns(self, usage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        return {
            'peak_usage_times': self._identify_peak_usage_times(usage_data),
            'usage_distribution': self._calculate_usage_distribution(usage_data),
            'correlation_analysis': self._analyze_usage_correlation(usage_data)
        }

    def _identify_peak_usage_times(self, usage_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify times of peak resource usage."""
        hour_usage = {}
        for data in usage_data:
            hour = datetime.fromisoformat(data['timestamp']).hour
            usage = data.get('utilization', 0)
            if hour not in hour_usage:
                hour_usage[hour] = []
            hour_usage[hour].append(usage)
        
        return [{
            'hour': hour,
            'average_usage': sum(usages) / len(usages),
            'peak_usage': max(usages)
        } for hour, usages in hour_usage.items()]

    def _calculate_usage_distribution(self, usage_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate distribution of resource usage."""
        if not usage_data:
            return {}
        
        usages = [data.get('utilization', 0) for data in usage_data]
        return {
            'min': min(usages),
            'max': max(usages),
            'average': sum(usages) / len(usages),
            'median': sorted(usages)[len(usages) // 2]
        }

    def _analyze_usage_correlation(self, usage_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze correlation between different resource metrics."""
        metrics = {}
        for data in usage_data:
            for metric, value in data.items():
                if isinstance(value, (int, float)) and metric != 'timestamp':
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].append(value)
        
        correlations = {}
        for m1 in metrics:
            for m2 in metrics:
                if m1 < m2:  # Avoid duplicate correlations
                    correlation = self._calculate_correlation(metrics[m1], metrics[m2])
                    correlations[f"{m1}-{m2}"] = correlation
        
        return correlations

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient between two metrics."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = (sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0.0

    # System Metrics Collection Methods
    def _get_cpu_usage(self) -> Dict[str, float]:
        """Get CPU usage metrics."""
        try:
            cpu_times = psutil.cpu_times_percent()
            return {
                'utilization': psutil.cpu_percent() / 100,
                'user': cpu_times.user,
                'system': cpu_times.system,
                'idle': cpu_times.idle
            }
        except Exception as e:
            self.logger.error(f"Error getting CPU usage: {str(e)}")
            return {'error': str(e)}

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage metrics."""
        try:
            memory = psutil.virtual_memory()
            return {
                'utilization': memory.percent / 100,
                'available': memory.available,
                'used': memory.used,
                'total': memory.total
            }
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {str(e)}")
            return {'error': str(e)}

    def _get_network_usage(self) -> Dict[str, float]:
        """Get network usage metrics."""
        try:
            network = psutil.net_io_counters()
            return {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        except Exception as e:
            self.logger.error(f"Error getting network usage: {str(e)}")
            return {'error': str(e)}

    def _get_storage_usage(self) -> Dict[str, float]:
        """Get storage usage metrics."""
        try:
            disk = psutil.disk_usage('/')
            return {
                'utilization': disk.percent / 100,
                'total': disk.total,
                'used': disk.used,
                'free': disk.free
            }
        except Exception as e:
            self.logger.error(f"Error getting storage usage: {str(e)}")
            return {'error': str(e)}

    def _get_agent_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """Get resource usage per agent."""
        usage = {}
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'get_resource_usage'):
                    usage[agent_name] = agent.get_resource_usage()
            except Exception as e:
                self.logger.error(f"Error getting resource usage for agent {agent_name}: {str(e)}")
        return usage

    def _get_queue_metrics(self) -> Dict[str, int]:
        """Get task queue metrics."""
        try:
            return {
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len([t for t in self.state.get('tasks', {}).values() 
                                   if t.get('status') == 'running']),
                'pending_tasks': len([t for t in self.state.get('tasks', {}).values() 
                                    if t.get('status') == 'pending'])
            }
        except Exception as e:
            self.logger.error(f"Error getting queue metrics: {str(e)}")
            return {'error': str(e)}

    def _get_cpu_max_frequency(self) -> float:
        """Get CPU maximum frequency."""
        try:
            return psutil.cpu_freq().max
        except Exception as e:
            self.logger.error(f"Error getting CPU max frequency: {str(e)}")
            return 0.0

    def _get_network_limits(self) -> Dict[str, Any]:
        """Get network interface limits."""
        try:
            interfaces = psutil.net_if_stats()
            return {
                name: {
                    'speed': interface.speed,
                    'mtu': interface.mtu,
                    'is_up': interface.isup
                }
                for name, interface in interfaces.items()
            }
        except Exception as e:
            self.logger.error(f"Error getting network limits: {str(e)}")
            return {'error': str(e)}

    def _predict_future_capacity(self) -> Dict[str, Any]:
        """Predict future system capacity needs."""
        try:
            current_usage = self._get_current_capacity()
            historical_data = self.state.get('historical_usage', [])
            
            predictions = {
                'cpu': self._predict_resource_usage(historical_data, 'cpu'),
                'memory': self._predict_resource_usage(historical_data, 'memory'),
                'storage': self._predict_resource_usage(historical_data, 'storage'),
                'network': self._predict_resource_usage(historical_data, 'network')
            }
            
            return {
                'current_usage': current_usage,
                'predictions': predictions,
                'recommendation': self._generate_capacity_recommendation(current_usage, predictions)
            }
        except Exception as e:
            self.logger.error(f"Error predicting future capacity: {str(e)}")
            return {'error': str(e)}

    def _predict_resource_usage(self, historical_data: List[Dict[str, Any]], resource_type: str) -> Dict[str, float]:
        """Predict future usage for a specific resource."""
        try:
            if not historical_data:
                return {}
            
            # Get usage trends
            usage_values = [data.get(resource_type, {}).get('utilization', 0) 
                          for data in historical_data]
            
            if not usage_values:
                return {}
            
            # Simple linear regression for prediction
            x = list(range(len(usage_values)))
            slope = self._calculate_slope(x, usage_values)
            
            # Predict next 24 hours
            current = usage_values[-1]
            predictions = {
                '1h': current + slope,
                '6h': current + slope * 6,
                '24h': current + slope * 24
            }
            
            return predictions
        except Exception as e:
            self.logger.error(f"Error predicting resource usage: {str(e)}")
            return {'error': str(e)}

    def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate slope for linear regression."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = sum((xi - mean_x) ** 2 for xi in x)
        
        return numerator / denominator if denominator != 0 else 0.0

    def _generate_capacity_recommendation(self, current_usage: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate capacity scaling recommendations."""
        try:
            recommendations = {}
            
            for resource in ['cpu', 'memory', 'storage', 'network']:
                current = current_usage.get(resource, {}).get('utilization', 0)
                predicted = predictions.get(resource, {}).get('24h', 0)
                
                if predicted > 0.8:  # 80% threshold
                    recommendations[resource] = {
                        'action': 'scale_up',
                        'urgency': 'high' if predicted > 0.9 else 'medium',
                        'reason': f"Predicted utilization of {predicted:.1%} exceeds threshold"
                    }
                elif predicted < 0.2:  # 20% threshold
                    recommendations[resource] = {
                        'action': 'scale_down',
                        'urgency': 'low',
                        'reason': f"Predicted utilization of {predicted:.1%} is below optimal"
                    }
                else:
                    recommendations[resource] = {
                        'action': 'maintain',
                        'urgency': 'none',
                        'reason': f"Predicted utilization of {predicted:.1%} is optimal"
                    }
            
            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating capacity recommendations: {str(e)}")
            return {'error': str(e)}

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        for agent in self.agents.values():
            agent.cleanup() 