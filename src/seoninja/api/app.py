"""Flask API for SEO Ninja."""
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import asyncio
from ..agents.orchestrator.orchestrator import SEOOrchestrator
from ..utils.storage import PersistentStorage
from ..utils.gsc_auth import get_gsc_credentials
import os
from dotenv import load_dotenv
from typing import Dict, Any
import json
from datetime import datetime
from functools import wraps

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize orchestrator
async def init_orchestrator():
    """Initialize orchestrator asynchronously."""
    load_dotenv()
    
    gsc_credentials = get_gsc_credentials(os.getenv('GSC_CLIENT_SECRETS_PATH', 'client_secrets.json'))
    moz_token = os.getenv('MOZ_API_TOKEN')
    domain = os.getenv('TARGET_DOMAIN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    storage = PersistentStorage()
    orchestrator = SEOOrchestrator(
        storage=storage,
        gsc_credentials=gsc_credentials,
        moz_token=moz_token,
        target_domain=domain,
        openai_api_key=openai_api_key
    )
    
    await orchestrator.run()
    return orchestrator

# Create event loop and initialize orchestrator
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
orchestrator = loop.run_until_complete(init_orchestrator())

def async_route(f):
    """Decorator to handle async routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return decorated_function

# Task Management Endpoints
@app.route('/api/tasks', methods=['POST'])
@async_route
async def create_task():
    """Create a new task for the agent system."""
    data = request.json
    task = {
        'type': data['type'],
        'parameters': data.get('parameters', {}),
        'priority': data.get('priority', 'normal'),
        'timestamp': datetime.now().isoformat()
    }
    
    result = await orchestrator.execute(task)
    return jsonify(result)

@app.route('/api/tasks/<task_id>', methods=['GET'])
@async_route
async def get_task_status(task_id: str):
    """Get status of a specific task."""
    task_status = orchestrator.state.get('tasks', {}).get(task_id, {})
    if not task_status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task_status)

@app.route('/api/tasks', methods=['GET'])
@async_route
async def list_tasks():
    """List all tasks with their status."""
    tasks = orchestrator.state.get('tasks', {})
    return jsonify(tasks)

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
@async_route
async def cancel_task(task_id: str):
    """Cancel a running task."""
    success = await orchestrator.cancel_task(task_id)
    if not success:
        return jsonify({'error': 'Task not found or cannot be cancelled'}), 404
    return jsonify({'status': 'cancelled'})

@app.route('/api/tasks/<task_id>/priority', methods=['PATCH'])
@async_route
async def update_task_priority(task_id: str):
    """Update task priority."""
    data = request.json
    new_priority = data.get('priority')
    if not new_priority:
        return jsonify({'error': 'Priority not specified'}), 400
    
    success = await orchestrator.update_task_priority(task_id, new_priority)
    if not success:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify({'status': 'updated'})

# Agent Management Endpoints
@app.route('/api/agents', methods=['GET'])
@async_route
async def list_agents():
    """List all available agents and their status."""
    agents = {
        name: {
            'status': agent.get_health(),
            'type': agent.__class__.__name__,
            'capabilities': [tool.name for tool in agent._tools]
        }
        for name, agent in orchestrator.agents.items()
    }
    return jsonify(agents)

@app.route('/api/agents/<agent_name>/health', methods=['GET'])
@async_route
async def get_agent_health(agent_name: str):
    """Get health status of a specific agent."""
    if agent_name not in orchestrator.agents:
        return jsonify({'error': 'Agent not found'}), 404
    
    health = await orchestrator.agents[agent_name].get_health()
    return jsonify(health)

# Real-time Updates Endpoint
@app.route('/api/updates/stream')
def stream_updates():
    """Stream real-time updates from the agent system."""
    def generate():
        while True:
            # Get latest updates from shared results
            updates = orchestrator.shared_results
            if updates:
                yield f"data: {json.dumps(updates)}\n\n"
            asyncio.sleep(1)
    
    return Response(generate(), mimetype='text/event-stream')

# Content Generation Endpoints
@app.route('/api/content/generate', methods=['POST'])
@async_route
async def generate_content():
    """Generate content endpoint."""
    data = request.json
    task = {
        'type': 'content_generation',
        'keyword': data['keyword'],
        'intent': data['intent'],
        'url': data.get('website_url')
    }
    
    result = await orchestrator.execute(task)
    return jsonify(result)

@app.route('/api/content/batch', methods=['POST'])
@async_route
async def batch_content_generation():
    """Batch content generation endpoint."""
    data = request.json
    results = []
    
    for keyword in data['keywords']:
        task = {
            'type': 'content_generation',
            'keyword': keyword,
            'intent': data['intent'],
            'url': data.get('website_url')
        }
        result = await orchestrator.execute(task)
        results.append(result)
    
    return jsonify(results)

# SEO Analysis Endpoints
@app.route('/api/seo/audit', methods=['POST'])
@async_route
async def run_seo_audit():
    """Run SEO audit endpoint."""
    data = request.json
    task = {
        'type': 'website_analysis',
        'url': data['url']
    }
    
    result = await orchestrator.execute(task)
    return jsonify(result)

@app.route('/api/seo/technical', methods=['POST'])
@async_route
async def analyze_technical_seo():
    """Analyze technical SEO aspects."""
    data = request.json
    task = {
        'type': 'technical_seo',
        'url': data['url']
    }
    
    result = await orchestrator.execute(task)
    return jsonify(result)

@app.route('/api/seo/recommendations', methods=['POST'])
@async_route
async def get_seo_recommendations():
    """Get SEO recommendations."""
    data = request.json
    task = {
        'type': 'recommendations',
        'url': data['url']
    }
    
    result = await orchestrator.execute(task)
    return jsonify(result)

# Keyword Research Endpoints
@app.route('/api/keywords/research', methods=['POST'])
@async_route
async def research_keywords():
    """Perform keyword research."""
    data = request.json
    task = {
        'type': 'keyword_research',
        'keyword': data['keyword'],
        'parameters': data.get('parameters', {})
    }
    
    result = await orchestrator.execute(task)
    return jsonify(result)

@app.route('/api/keywords/analyze', methods=['POST'])
@async_route
async def analyze_keywords():
    """Analyze keywords for a given content."""
    data = request.json
    task = {
        'type': 'keyword_analysis',
        'content': data['content'],
        'keywords': data.get('keywords', [])
    }
    
    result = await orchestrator.execute(task)
    return jsonify(result)

# System Status and Management
@app.route('/api/system/status', methods=['GET'])
@async_route
async def get_system_status():
    """Get overall system status."""
    status = {
        'agents': {name: agent.get_health() for name, agent in orchestrator.agents.items()},
        'tasks': len(orchestrator.state.get('tasks', {})),
        'active_tasks': len([t for t in orchestrator.state.get('tasks', {}).values() if t.get('status') == 'running']),
        'system_health': orchestrator.get_health(),
        'last_update': datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/api/system/metrics', methods=['GET'])
@async_route
async def get_system_metrics():
    """Get system performance metrics."""
    metrics = {
        'task_completion_rate': orchestrator.get_task_completion_rate(),
        'average_response_time': orchestrator.get_average_response_time(),
        'error_rate': orchestrator.get_error_rate(),
        'agent_performance': {
            name: agent.get_performance_metrics()
            for name, agent in orchestrator.agents.items()
        }
    }
    return jsonify(metrics)

# Historical Data and Analytics
@app.route('/api/analytics/performance', methods=['GET'])
@async_route
async def get_performance_history():
    """Get historical performance data."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    metrics = await orchestrator.get_historical_metrics(start_date, end_date)
    return jsonify(metrics)

@app.route('/api/analytics/tasks', methods=['GET'])
@async_route
async def get_task_history():
    """Get historical task data."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    task_type = request.args.get('type')
    history = await orchestrator.get_task_history(start_date, end_date, task_type)
    return jsonify(history)

# Configuration Management
@app.route('/api/config', methods=['GET'])
@async_route
async def get_configuration():
    """Get current system configuration."""
    config = await orchestrator.get_configuration()
    # Mask sensitive data
    if 'api_keys' in config:
        config['api_keys'] = {k: '***' for k in config['api_keys']}
    return jsonify(config)

@app.route('/api/config', methods=['PATCH'])
@async_route
async def update_configuration():
    """Update system configuration."""
    data = request.json
    success = await orchestrator.update_configuration(data)
    return jsonify({'status': 'updated' if success else 'failed'})

# Agent Control
@app.route('/api/agents/<agent_name>/pause', methods=['POST'])
@async_route
async def pause_agent(agent_name: str):
    """Pause a specific agent."""
    if agent_name not in orchestrator.agents:
        return jsonify({'error': 'Agent not found'}), 404
    success = await orchestrator.pause_agent(agent_name)
    return jsonify({'status': 'paused' if success else 'failed'})

@app.route('/api/agents/<agent_name>/resume', methods=['POST'])
@async_route
async def resume_agent(agent_name: str):
    """Resume a paused agent."""
    if agent_name not in orchestrator.agents:
        return jsonify({'error': 'Agent not found'}), 404
    success = await orchestrator.resume_agent(agent_name)
    return jsonify({'status': 'resumed' if success else 'failed'})

@app.route('/api/agents/<agent_name>/reset', methods=['POST'])
@async_route
async def reset_agent(agent_name: str):
    """Reset a specific agent to its initial state."""
    if agent_name not in orchestrator.agents:
        return jsonify({'error': 'Agent not found'}), 404
    success = await orchestrator.reset_agent(agent_name)
    return jsonify({'status': 'reset' if success else 'failed'})

# Error Logging and Debugging
@app.route('/api/logs', methods=['GET'])
@async_route
async def get_system_logs():
    """Get system logs."""
    log_level = request.args.get('level', 'INFO')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    limit = request.args.get('limit', 100)
    
    logs = await orchestrator.get_logs(log_level, start_time, end_time, limit)
    return jsonify(logs)

@app.route('/api/logs/errors', methods=['GET'])
@async_route
async def get_error_logs():
    """Get error logs specifically."""
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    limit = request.args.get('limit', 50)
    
    errors = await orchestrator.get_error_logs(start_time, end_time, limit)
    return jsonify(errors)

@app.route('/api/debug/state', methods=['GET'])
@async_route
async def get_debug_state():
    """Get detailed system state for debugging."""
    include_sensitive = request.args.get('include_sensitive', 'false').lower() == 'true'
    state = await orchestrator.get_debug_state(include_sensitive)
    return jsonify(state)

if __name__ == '__main__':
    app.run(debug=True) 