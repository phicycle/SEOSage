"""Utility for parallel task execution."""
import asyncio
from typing import List, Dict, Any
from ..agents.base.agent import BaseAgent

async def run_parallel_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run multiple agent tasks in parallel.
    
    Args:
        tasks: List of task dictionaries, each containing:
            - agent: BaseAgent instance
            - task: Task dictionary for the agent
            
    Returns:
        List of results from each task
    """
    async def execute_task(task_dict: Dict[str, Any]) -> Dict[str, Any]:
        agent: BaseAgent = task_dict['agent']
        task_data: Dict[str, Any] = task_dict['task']
        return await agent.execute(task_data)
        
    # Create coroutines for each task
    coroutines = [execute_task(task) for task in tasks]
    
    # Run tasks in parallel and return results
    return await asyncio.gather(*coroutines) 