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
from flask_apispec import use_kwargs, marshal_with
from flask_apispec.views import MethodResource
from marshmallow import Schema, fields

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

# Schema definitions for request/response serialization
class TaskCreateSchema(Schema):
    """Schema for task creation request."""
    type = fields.Str(required=True, description="Type of task to create")
    parameters = fields.Dict(description="Task parameters")
    priority = fields.Str(description="Task priority level")

class TaskResponseSchema(Schema):
    """Schema for task creation response."""
    success = fields.Bool()
    task_id = fields.Str()
    status = fields.Str()

class TaskStatusSchema(Schema):
    """Schema for task status response."""
    task_id = fields.Str(required=True)
    status = fields.Str()
    progress = fields.Float()
    timestamp = fields.DateTime()

class AgentHealthSchema(Schema):
    """Schema for agent health status."""
    status = fields.Str()
    last_check = fields.DateTime()
    metrics = fields.Dict()

class LogQuerySchema(Schema):
    """Schema for log query parameters."""
    level = fields.Str(missing="INFO")
    start_time = fields.DateTime(required=False)
    end_time = fields.DateTime(required=False)
    limit = fields.Int(missing=100)

class CommunicationFlowSchema(Schema):
    """Schema for communication flow data."""
    source = fields.Str()
    target = fields.Str()
    message_type = fields.Str()
    timestamp = fields.DateTime()

class ProcessTimelineSchema(Schema):
    """Schema for process timeline data."""
    process_id = fields.Str(required=True)
    stages = fields.List(fields.Dict())
    current_stage = fields.Str()
    duration = fields.Float()
    status = fields.Str()

class ResourceMetricsSchema(Schema):
    """Schema for resource metrics data."""
    resource_type = fields.Str()
    utilization = fields.Float()
    timestamp = fields.DateTime()
    metrics = fields.Dict()

class ResourceAllocationSchema(Schema):
    """Schema for resource allocation data."""
    resource_type = fields.Str()
    timeframe = fields.Str()
    allocations = fields.List(fields.Dict())
    metrics = fields.Dict()

class ResourceUsageSchema(Schema):
    """Schema for resource usage patterns."""
    patterns = fields.Dict()
    trends = fields.Dict()
    predictions = fields.Dict()

class ResourceBottleneckSchema(Schema):
    """Schema for resource bottleneck data."""
    bottlenecks = fields.List(fields.Dict())
    impact_analysis = fields.Dict()
    recommendations = fields.Dict()

class SystemCapacitySchema(Schema):
    """Schema for system capacity metrics."""
    current_capacity = fields.Dict()
    utilization = fields.Dict()
    headroom = fields.Dict()
    limits = fields.Dict()
    predictions = fields.Dict(required=False)

# Task Management Endpoints
@app.route('/api/tasks', methods=['POST'])
@use_kwargs(TaskCreateSchema)
@marshal_with(TaskResponseSchema)
async def create_task():
    """Create a new task for the agent system.
    
    This endpoint creates a new task in the system. The task will be queued and processed
    by the appropriate agents based on its type and parameters.
    
    ---
    post:
      description: Create a new task
      parameters:
        - in: body
          name: body
          schema:
            type: object
            required:
              - type
            properties:
              type:
                type: string
                description: Type of task to create
              parameters:
                type: object
                description: Task parameters
              priority:
                type: string
                description: Task priority level
      responses:
        200:
          description: Task created successfully
          schema:
            type: object
            properties:
              success:
                type: boolean
              task_id:
                type: string
              status:
                type: string
        400:
          description: Invalid parameters
    """
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
    tasks = orchestrator.state.get('tasks', [])
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
    agents = {}
    for name, agent in orchestrator.agents.items():
        agent_info = {
            'status': await agent.get_health(),
            'type': agent.__class__.__name__,
        }
        
        # Safely get capabilities
        if hasattr(agent, '_tools'):
            agent_info['capabilities'] = [tool.name for tool in agent._tools]
        elif hasattr(agent, '_setup_tools'):
            # For agents that set up tools differently
            tools = agent._setup_tools()
            agent_info['capabilities'] = [tool.name for tool in tools] if tools else []
        else:
            agent_info['capabilities'] = []
            
        agents[name] = agent_info
    
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
    # Get agent health statuses concurrently
    agent_healths = {}
    for name, agent in orchestrator.agents.items():
        agent_healths[name] = await agent.get_health()
    
    status = {
        'agents': agent_healths,
        'tasks': len(orchestrator.state.get('tasks', {})),
        'active_tasks': len([t for t in orchestrator.state.get('tasks', {}).values() if t.get('status') == 'running']),
        'system_health': await orchestrator.get_health(),
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

# Agent Communication Visualization Endpoints
@app.route('/api/communication/flows', methods=['GET'])
@use_kwargs({
    'start_time': fields.DateTime(required=False),
    'end_time': fields.DateTime(required=False)
}, location='query')
@marshal_with(CommunicationFlowSchema(many=True))
async def get_communication_flows(start_time: datetime = None, end_time: datetime = None):
    """Get inter-agent communication flows.
    ---
    get:
      description: Get communication flows between agents
      parameters:
        - in: query
          name: start_time
          type: string
          format: date-time
          description: Start time for filtering flows
        - in: query
          name: end_time
          type: string
          format: date-time
          description: End time for filtering flows
      responses:
        200:
          description: Communication flows retrieved successfully
          schema:
            type: array
            items:
              $ref: '#/definitions/CommunicationFlowSchema'
    """
    flows = await orchestrator.get_communication_flows(start_time, end_time)
    return jsonify(flows)

@app.route('/api/communication/realtime', methods=['GET'])
def stream_communication_data():
    """Stream real-time agent interaction data."""
    def generate():
        while True:
            comm_data = orchestrator.get_realtime_communication()
            if comm_data:
                yield f"data: {json.dumps(comm_data)}\n\n"
            asyncio.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/communication/analytics', methods=['GET'])
@use_kwargs({'timeframe': fields.Str(missing='24h')}, location='query')
async def get_communication_analytics(timeframe: str = '24h'):
    """Get communication patterns analytics.
    ---
    get:
      description: Get analytics about communication patterns between agents
      parameters:
        - in: query
          name: timeframe
          type: string
          default: 24h
          description: Time frame for analysis (e.g., '24h', '7d')
      responses:
        200:
          description: Communication analytics retrieved successfully
          schema:
            type: object
            properties:
              patterns:
                type: object
              metrics:
                type: object
              insights:
                type: object
    """
    metrics = await orchestrator.get_communication_analytics(timeframe)
    return jsonify(metrics)

@app.route('/api/communication/metrics', methods=['GET'])
@async_route
async def get_data_flow_metrics():
    """Get data flow metrics between agents."""
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    metrics = await orchestrator.get_data_flow_metrics(start_time, end_time)
    return jsonify(metrics)

# Process Timeline Endpoints
@app.route('/api/timeline/process', methods=['GET'])
@use_kwargs({
    'process_id': fields.Str(required=True),
    'include_details': fields.Bool(missing=False)
}, location='query')
@marshal_with(ProcessTimelineSchema)
async def get_process_timeline(process_id: str, include_details: bool = False):
    """Get detailed process stage tracking data.
    ---
    get:
      description: Get timeline and stages of a specific process
      parameters:
        - in: query
          name: process_id
          type: string
          required: true
          description: ID of the process to track
        - in: query
          name: include_details
          type: boolean
          default: false
          description: Whether to include additional stage details
      responses:
        200:
          description: Process timeline retrieved successfully
          schema:
            $ref: '#/definitions/ProcessTimelineSchema'
        404:
          description: Process not found
    """
    timeline = await orchestrator.get_process_timeline(process_id, include_details)
    if not timeline:
        return jsonify({'error': 'Process not found'}), 404
    return jsonify(timeline)

@app.route('/api/timeline/decisions', methods=['GET'])
@use_kwargs({
    'process_id': fields.Str(required=True),
    'start_time': fields.DateTime(required=False),
    'end_time': fields.DateTime(required=False)
}, location='query')
async def get_decision_points(process_id: str, start_time: datetime = None, end_time: datetime = None):
    """Get decision point logging data.
    ---
    get:
      description: Get decision points and their outcomes for a process
      parameters:
        - in: query
          name: process_id
          type: string
          required: true
          description: ID of the process
        - in: query
          name: start_time
          type: string
          format: date-time
          description: Start time for filtering decisions
        - in: query
          name: end_time
          type: string
          format: date-time
          description: End time for filtering decisions
      responses:
        200:
          description: Decision points retrieved successfully
          schema:
            type: object
            properties:
              decisions:
                type: array
                items:
                  type: object
              metrics:
                type: object
    """
    decisions = await orchestrator.get_decision_points(process_id, start_time, end_time)
    return jsonify(decisions)

@app.route('/api/timeline/transitions', methods=['GET'])
@use_kwargs({'process_id': fields.Str(required=True)}, location='query')
async def get_stage_transitions(process_id: str):
    """Get stage transition data.
    ---
    get:
      description: Get stage transitions for a process
      parameters:
        - in: query
          name: process_id
          type: string
          required: true
          description: ID of the process
      responses:
        200:
          description: Stage transitions retrieved successfully
          schema:
            type: object
            properties:
              transitions:
                type: array
                items:
                  type: object
              metrics:
                type: object
    """
    transitions = await orchestrator.get_stage_transitions(process_id)
    return jsonify(transitions)

@app.route('/api/timeline/analysis', methods=['GET'])
@use_kwargs({
    'timeframe': fields.Str(missing='24h'),
    'process_type': fields.Str(required=False)
}, location='query')
async def get_time_analysis(timeframe: str = '24h', process_type: str = None):
    """Get time-based process analysis.
    ---
    get:
      description: Get time-based analysis of processes
      parameters:
        - in: query
          name: timeframe
          type: string
          default: 24h
          description: Time frame for analysis (e.g., '24h', '7d')
        - in: query
          name: process_type
          type: string
          description: Type of process to analyze
      responses:
        200:
          description: Time analysis retrieved successfully
          schema:
            type: object
            properties:
              metrics:
                type: object
              trends:
                type: object
              optimization_suggestions:
                type: object
    """
    analysis = await orchestrator.get_time_analysis(timeframe, process_type)
    return jsonify(analysis)

# Resource Monitoring Endpoints
@app.route('/api/resources/allocation', methods=['GET'])
@use_kwargs({'resource_type': fields.Str(), 'timeframe': fields.Str(missing='1h')}, location='query')
@marshal_with(ResourceAllocationSchema)
async def get_resource_allocation(resource_type: str = None, timeframe: str = '1h'):
    """Get detailed resource allocation tracking.
    ---
    get:
      description: Get resource allocation metrics
      parameters:
        - in: query
          name: resource_type
          type: string
          description: Type of resource to monitor
        - in: query
          name: timeframe
          type: string
          default: 1h
          description: Time frame for analysis (e.g., '1h', '24h')
      responses:
        200:
          description: Resource allocation metrics retrieved successfully
          schema:
            $ref: '#/definitions/ResourceAllocationSchema'
    """
    allocation = await orchestrator.get_resource_allocation(resource_type, timeframe)
    return jsonify(allocation)

@app.route('/api/resources/usage-patterns', methods=['GET'])
@use_kwargs({
    'resource_type': fields.Str(),
    'start_time': fields.DateTime(required=False),
    'end_time': fields.DateTime(required=False)
}, location='query')
@marshal_with(ResourceUsageSchema)
async def get_usage_patterns(resource_type: str = None, start_time: datetime = None, end_time: datetime = None):
    """Get resource usage patterns.
    ---
    get:
      description: Get resource usage patterns and trends
      parameters:
        - in: query
          name: resource_type
          type: string
          description: Type of resource to analyze
        - in: query
          name: start_time
          type: string
          format: date-time
          description: Start time for analysis
        - in: query
          name: end_time
          type: string
          format: date-time
          description: End time for analysis
      responses:
        200:
          description: Resource usage patterns retrieved successfully
          schema:
            $ref: '#/definitions/ResourceUsageSchema'
    """
    patterns = await orchestrator.get_usage_patterns(resource_type, start_time, end_time)
    return jsonify(patterns)

@app.route('/api/resources/bottlenecks', methods=['GET'])
@use_kwargs({
    'analysis_period': fields.Str(missing='24h'),
    'threshold': fields.Float(missing=0.8)
}, location='query')
@marshal_with(ResourceBottleneckSchema)
async def get_resource_bottlenecks(analysis_period: str = '24h', threshold: float = 0.8):
    """Get resource bottleneck identification.
    ---
    get:
      description: Identify resource bottlenecks and get recommendations
      parameters:
        - in: query
          name: analysis_period
          type: string
          default: 24h
          description: Period for analysis (e.g., '24h', '7d')
        - in: query
          name: threshold
          type: number
          default: 0.8
          description: Utilization threshold for bottleneck detection (0.0-1.0)
      responses:
        200:
          description: Resource bottlenecks identified successfully
          schema:
            $ref: '#/definitions/ResourceBottleneckSchema'
    """
    bottlenecks = await orchestrator.get_resource_bottlenecks(analysis_period, threshold)
    return jsonify(bottlenecks)

@app.route('/api/resources/capacity', methods=['GET'])
@use_kwargs({'include_predictions': fields.Bool(missing=False)}, location='query')
@marshal_with(SystemCapacitySchema)
async def get_system_capacity(include_predictions: bool = False):
    """Get system capacity metrics.
    ---
    get:
      description: Get current system capacity and optional future predictions
      parameters:
        - in: query
          name: include_predictions
          type: boolean
          default: false
          description: Whether to include capacity predictions
      responses:
        200:
          description: System capacity metrics retrieved successfully
          schema:
            $ref: '#/definitions/SystemCapacitySchema'
    """
    capacity = await orchestrator.get_system_capacity(include_predictions)
    return jsonify(capacity)

@app.route('/api/resources/realtime', methods=['GET'])
def stream_resource_metrics():
    """Stream real-time resource metrics.
    ---
    get:
      description: Get real-time resource metrics as Server-Sent Events
      responses:
        200:
          description: SSE stream started
          content:
            text/event-stream:
              schema:
                type: object
                properties:
                  timestamp:
                    type: string
                    format: date-time
                  cpu_usage:
                    type: object
                  memory_usage:
                    type: object
                  network_usage:
                    type: object
                  agent_resources:
                    type: object
                  queue_metrics:
                    type: object
    """
    def generate():
        while True:
            metrics = orchestrator.get_realtime_resource_metrics()
            if metrics:
                yield f"data: {json.dumps(metrics)}\n\n"
            asyncio.sleep(1)
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True) 