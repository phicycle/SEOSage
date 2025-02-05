"""Base agent class for SEO Ninja."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Callable
import logging
from datetime import datetime, timedelta
from ...config.settings import get_settings

class BaseAgent(ABC):
    """Base agent class with core functionality."""
    
    def __init__(self, name: str):
        """Initialize base agent."""
        self.name = name
        self.logger = logging.getLogger(f"seoninja.agent.{name}")
        self.memory: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {}
        self.collaborators: Set[str] = set()
        self.shared_results: Dict[str, Any] = {}
        self.memory_max_age = timedelta(hours=1)  # Default memory retention
        self.memory_max_size = 1000  # Maximum memory entries
        self.observers = []
        self.handlers = {}
        
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's primary task."""
        pass
        
    def cleanup_memory(self, max_age: Optional[timedelta] = None) -> None:
        """Clean up old memory entries."""
        cutoff = datetime.now() - (max_age or self.memory_max_age)
        self.memory = [m for m in self.memory if datetime.fromisoformat(m['timestamp']) > cutoff]
        
        # Enforce maximum memory size
        if len(self.memory) > self.memory_max_size:
            self.memory = sorted(
                self.memory,
                key=lambda x: datetime.fromisoformat(x['timestamp']),
                reverse=True
            )[:self.memory_max_size]
        
    def remember(self, observation: Dict[str, Any]) -> None:
        """Store observation in agent's memory with enhanced metadata."""
        observation.update({
            'timestamp': datetime.now().isoformat(),
            'agents_involved': observation.get('agents_involved', [self.name]),
            'related_tasks': observation.get('related_tasks', []),
            'success_metrics': observation.get('success_metrics', {}),
            'context': observation.get('context', {}),
            'priority': observation.get('priority', 'normal')
        })
        
        self.memory.append(observation)
        self._broadcast_observation(observation)
        
        # Cleanup old entries after adding new one
        self.cleanup_memory()
        
    def _broadcast_observation(self, observation: Dict[str, Any]) -> None:
        """Share relevant observations with collaborating agents."""
        if not observation.get('private', False):
            for collaborator in self.collaborators:
                self.shared_results[collaborator] = {
                    'type': 'observation',
                    'content': observation,
                    'source': self.name,
                    'timestamp': datetime.now().isoformat()
                }
        
    def recall(self, 
              observation_type: str = None, 
              time_range: tuple = None, 
              agents: List[str] = None,
              priority: str = None) -> List[Dict[str, Any]]:
        """Enhanced memory retrieval with filtering capabilities."""
        filtered_memory = self.memory
        
        if observation_type:
            filtered_memory = [obs for obs in filtered_memory if obs['type'] == observation_type]
            
        if time_range:
            start, end = time_range
            filtered_memory = [
                obs for obs in filtered_memory 
                if start <= obs['timestamp'] <= end
            ]
            
        if agents:
            filtered_memory = [
                obs for obs in filtered_memory 
                if any(agent in obs['agents_involved'] for agent in agents)
            ]
            
        if priority:
            filtered_memory = [obs for obs in filtered_memory if obs['priority'] == priority]
            
        return filtered_memory
        
    def _filter_relevant_memory(self, task_type: str) -> List[Dict[str, Any]]:
        """Filter memory relevant to specific task type."""
        return [
            mem for mem in self.memory 
            if task_type in mem.get('related_tasks', []) 
            or mem.get('type') == task_type
        ]
        
    def _filter_relevant_state(self, task_type: str) -> Dict[str, Any]:
        """Filter state relevant to specific task type."""
        relevant_keys = [k for k in self.state.keys() if task_type in k]
        return {k: self.state[k] for k in relevant_keys}
        
    async def collaborate(self, agent: 'BaseAgent', task: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced collaboration with context sharing and feedback."""
        try:
            # Add collaborator
            self.collaborators.add(agent.name)
            agent.collaborators.add(self.name)
            
            # Prepare relevant context
            relevant_memory = self._filter_relevant_memory(task['type'])
            relevant_state = self._filter_relevant_state(task['type'])
            
            shared_context = {
                'requester': self.name,
                'task_type': task['type'],
                'relevant_state': relevant_state,
                'relevant_memory': relevant_memory,
                'dependencies': task.get('dependencies', []),
                'previous_results': task.get('previous_results', {}),
                'collaboration_chain': task.get('collaboration_chain', []) + [self.name]
            }
            
            task['context'] = shared_context
            
            # Execute collaboration
            result = await agent.execute(task)
            
            # Process feedback
            if result.get('feedback'):
                await self._process_feedback(result['feedback'])
            
            # Update state with results
            if result.get('success'):
                self.update_state({
                    f'collaboration_{agent.name}_{task["type"]}': result.get('data')
                })
                
            # Remember collaboration
            self.remember({
                'type': 'collaboration',
                'agents_involved': [self.name, agent.name],
                'task_type': task['type'],
                'success': result.get('success', False),
                'timestamp': datetime.now().isoformat()
            })
                
            return result
            
        except Exception as e:
            self.logger.error(f"Collaboration failed with {agent.name}: {str(e)}")
            return {
                'success': False,
                'error': f"Collaboration failed: {str(e)}"
            }
            
    async def _process_feedback(self, feedback: Dict[str, Any]) -> None:
        """Process feedback from collaboration."""
        if feedback.get('state_updates'):
            self.update_state(feedback['state_updates'])
            
        if feedback.get('memory_updates'):
            for observation in feedback['memory_updates']:
                self.remember(observation)
                
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update agent's current state with timestamp."""
        updates = {
            k: {
                'value': v,
                'updated_at': datetime.now().isoformat(),
                'updated_by': self.name
            } for k, v in updates.items()
        }
        self.state.update(updates)
        
    def get_state(self, key: str = None, include_metadata: bool = False) -> Any:
        """Get agent's current state with optional metadata."""
        if key:
            state_item = self.state.get(key)
            return state_item if include_metadata else state_item['value'] if state_item else None
        return self.state
        
    def validate_task(self, task: Dict[str, Any], required_fields: List[str]) -> Optional[str]:
        """Validate task has required fields."""
        missing = [field for field in required_fields if field not in task]
        if missing:
            return f"Missing required fields: {', '.join(missing)}"
        return None
        
    def log_progress(self, message: str, level: str = 'info') -> None:
        """Log progress with appropriate level."""
        log_method = getattr(self.logger, level.lower())
        log_method(f"[{self.name}] {message}")
        
    def cleanup(self) -> None:
        """Clean up resources and notify collaborators."""
        # Notify collaborators of cleanup
        for collaborator in self.collaborators:
            self.shared_results[collaborator] = {
                'type': 'cleanup',
                'source': self.name,
                'timestamp': datetime.now().isoformat()
            }
            
        self.memory.clear()
        self.state.clear()
        self.collaborators.clear()
        self.shared_results.clear()
        
        self.observers.clear()
        self.handlers.clear()
        
    def add_observer(self, observer: Any) -> None:
        """Add an observer to this agent."""
        if observer not in self.observers:
            self.observers.append(observer)
            
    def remove_observer(self, observer: Any) -> None:
        """Remove an observer from this agent."""
        if observer in self.observers:
            self.observers.remove(observer)
            
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for a specific event type."""
        self.handlers[event_type] = handler
        
    async def notify_observers(self, event: Dict[str, Any]) -> None:
        """Notify all observers of an event."""
        for observer in self.observers:
            try:
                await observer.handle_event(self.name, event)
            except Exception as e:
                self.logger.error(f"Error notifying observer {observer}: {str(e)}")
                
    async def update_context(self, context: Dict[str, Any]) -> None:
        """Update agent's context."""
        try:
            self.state.update(context)
        except Exception as e:
            self.logger.error(f"Error updating context: {str(e)}")
            
    async def get_health(self) -> Dict[str, Any]:
        """Get agent's health status."""
        return {
            'healthy': True,
            'status': 'operational',
            'last_check': datetime.now().isoformat()
        } 