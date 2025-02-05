# SEO Ninja - Intelligent Agent System

SEO Ninja is a sophisticated multi-agent system designed for comprehensive SEO analysis, content generation, and optimization. The system utilizes multiple specialized agents orchestrated by a central coordinator to perform complex SEO tasks efficiently.

## 🚀 System Architecture

### Core Components

1. **Orchestrator Agent**
   - Central coordinator managing task decomposition and execution
   - Handles inter-agent communication and state management
   - Provides real-time monitoring and health checks
   - Manages task queues and priorities

2. **Specialized Agents**
   - **Content Agent**: Generates and optimizes content
   - **SEO Optimizer Agent**: Handles technical SEO analysis and recommendations
   - **Crawler Agent**: Performs website crawling and structure analysis
   - **Keyword Research Agent**: Conducts keyword research and analysis

### System Flow

The SEO Ninja system operates through a sophisticated multi-agent architecture with several key workflows:

#### 1. Task Lifecycle
```
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  User Input  │ -> │ API Gateway  │ -> │   Orchestrator  │ -> │ Task Queue   │
└──────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
                                                │
                                                ▼
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐    
│   Response   │ <- │  Aggregator  │ <- │ Agent Execution │    
└──────────────┘    └──────────────┘    └─────────────────┘    
```

**Detailed Flow:**
1. **User Input Processing**
   - Request validation and sanitization
   - Authentication and rate limiting
   - Task priority assignment

2. **Orchestrator Processing**
   - Task decomposition into subtasks
   - Resource allocation
   - Agent selection and scheduling

3. **Execution Phase**
   - Parallel/Sequential task execution
   - Inter-agent communication
   - Progress monitoring

4. **Result Aggregation**
   - Data consolidation
   - Format standardization
   - Response preparation

#### 2. Agent Interaction Model
```
                        ┌─────────────────────┐
                        │     Orchestrator    │
                        └─────────────────────┘
                           │      │      │
                           ▼      ▼      ▼
┌──────────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│Content Agent │<-->│SEO Agent │<-->│Crawler   │<-->│Keyword    │
└──────────────┘    └──────────┘    │Agent    │    │Agent      │
        ▲                ▲          └──────────┘    └───────────┘
        │                │               ▲              ▲
        │                │               │              │
        └────────────────┴───────────────┴──────────────┘
             Shared Context & Results Pool
```

**Communication Patterns:**
1. **Vertical Communication (Orchestrator ↔ Agents)**
   - Task assignments
   - Status updates
   - Resource allocation
   - Priority adjustments

2. **Horizontal Communication (Agent ↔ Agent)**
   - Shared context updates
   - Intermediate results sharing
   - Dependency resolution
   - Task synchronization

#### 3. Real-time Updates Flow
```
┌──────────────┐    ┌───────────────┐    ┌────────────┐    ┌──────────┐
│Agent Actions │ -> │Shared Results │ -> │SSE Stream  │ -> │UI Client │
└──────────────┘    └───────────────┘    └────────────┘    └──────────┘
       │                    ▲                   │               │
       └────────────────────┘                   └───────────────┘
           Real-time Updates                    Event Handling
```

**Update Types:**
1. **Progress Updates**
   - Task status changes
   - Completion percentages
   - Stage transitions

2. **Result Updates**
   - Intermediate findings
   - Partial completions
   - Error notifications

3. **System Updates**
   - Agent health status
   - Resource utilization
   - Performance metrics

#### 4. Task Decomposition Example
```
Content Generation Task
│
├── Keyword Research
│   ├── Seed keyword analysis
│   ├── Competition research
│   └── Intent mapping
│
├── Content Creation
│   ├── Outline generation
│   ├── Content writing
│   └── Media suggestions
│
└── Optimization
    ├── Technical SEO check
    ├── Content optimization
    └── Performance analysis
```

#### 5. State Management
```
┌─────────────────┐
│  Global State   │
├─────────────────┤
│ - Task Queue    │
│ - Agent States  │
│ - Shared Context│
└─────────────────┘
        │
        ▼
┌─────────────────┐    ┌─────────────────┐
│  Task State     │    │  Agent State    │
├─────────────────┤    ├─────────────────┤
│ - Progress      │    │ - Health        │
│ - Dependencies  │    │ - Capacity      │
│ - Results       │    │ - Tasks         │
└─────────────────┘    └─────────────────┘
```

#### 6. Error Handling Flow
```
┌──────────────┐    ┌───────────────┐    ┌────────────────┐
│Error Detected│ -> │Error Analysis │ -> │Recovery Action │
└──────────────┘    └───────────────┘    └────────────────┘
                           │                      │
                           ▼                      ▼
                    ┌───────────────┐    ┌────────────────┐
                    │User Notification│   │State Restoration│
                    └───────────────┘    └────────────────┘
```

**Error Handling Strategies:**
1. **Task Level**
   - Retry mechanisms
   - Fallback options
   - Partial results handling

2. **Agent Level**
   - Health monitoring
   - Auto-recovery
   - Load balancing

3. **System Level**
   - Circuit breaking
   - Graceful degradation
   - Resource reallocation

## 🔧 Technical Setup

### Prerequisites

- Python 3.8+
- Flask
- aiohttp
- OpenAI API key
- Moz API token (for SEO analysis)
- Google Search Console credentials

### Environment Variables

```env
GSC_CLIENT_SECRETS_PATH=client_secrets.json
MOZ_API_TOKEN=your_moz_token
TARGET_DOMAIN=your_domain
OPENAI_API_KEY=your_openai_key
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/seoninja.git

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m src.seoninja.api.app
```

## 📡 API Endpoints

### Task Management

#### Create Task
```http
POST /api/tasks
Content-Type: application/json

{
    "type": "content_generation",
    "parameters": {
        "keyword": "example keyword",
        "intent": "informational"
    },
    "priority": "high"
}
```

#### Get Task Status
```http
GET /api/tasks/{task_id}
```

#### List Tasks
```http
GET /api/tasks
```

#### Cancel Task
```http
DELETE /api/tasks/{task_id}
```

#### Update Task Priority
```http
PATCH /api/tasks/{task_id}/priority
Content-Type: application/json

{
    "priority": "high"
}
```

### Agent Management

#### List Agents
```http
GET /api/agents
```

#### Get Agent Health
```http
GET /api/agents/{agent_name}/health
```

#### Pause Agent
```http
POST /api/agents/{agent_name}/pause
```

#### Resume Agent
```http
POST /api/agents/{agent_name}/resume
```

#### Reset Agent
```http
POST /api/agents/{agent_name}/reset
```

### Analytics and Monitoring

#### Get Performance History
```http
GET /api/analytics/performance?start_date=2024-01-01&end_date=2024-01-31
```

#### Get Task History
```http
GET /api/analytics/tasks?start_date=2024-01-01&end_date=2024-01-31&type=content_generation
```

#### Get System Metrics
```http
GET /api/system/metrics
```

#### Get System Status
```http
GET /api/system/status
```

### Configuration Management

#### Get Configuration
```http
GET /api/config
```

#### Update Configuration
```http
PATCH /api/config
Content-Type: application/json

{
    "setting_name": "setting_value"
}
```

### Logging and Debugging

#### Get System Logs
```http
GET /api/logs?level=INFO&start_time=2024-01-01T00:00:00&end_time=2024-01-31T23:59:59&limit=100
```

#### Get Error Logs
```http
GET /api/logs/errors?start_time=2024-01-01T00:00:00&end_time=2024-01-31T23:59:59&limit=50
```

#### Get Debug State
```http
GET /api/debug/state?include_sensitive=false
```

### Real-time Updates

#### Stream Updates
```http
GET /api/updates/stream
```
Response: Server-Sent Events (SSE) stream

### Content Generation

#### Generate Content
```http
POST /api/content/generate
Content-Type: application/json

{
    "keyword": "target keyword",
    "intent": "informational",
    "website_url": "https://example.com"
}
```

#### Batch Content Generation
```http
POST /api/content/batch
Content-Type: application/json

{
    "keywords": ["keyword1", "keyword2"],
    "intent": "informational",
    "website_url": "https://example.com"
}
```

### SEO Analysis

#### Run SEO Audit
```http
POST /api/seo/audit
Content-Type: application/json

{
    "url": "https://example.com"
}
```

#### Analyze Technical SEO
```http
POST /api/seo/technical
Content-Type: application/json

{
    "url": "https://example.com"
}
```

#### Get SEO Recommendations
```http
POST /api/seo/recommendations
Content-Type: application/json

{
    "url": "https://example.com"
}
```

### Keyword Research

#### Research Keywords
```http
POST /api/keywords/research
Content-Type: application/json

{
    "keyword": "seed keyword",
    "parameters": {
        "locale": "en-US",
        "volume_threshold": 1000
    }
}
```

#### Analyze Keywords
```http
POST /api/keywords/analyze
Content-Type: application/json

{
    "content": "Your content here",
    "keywords": ["keyword1", "keyword2"]
}
```

## 🔄 Real-time Updates Integration

```javascript
const eventSource = new EventSource('/api/updates/stream');

eventSource.onmessage = (event) => {
    const updates = JSON.parse(event.data);
    // Handle updates in UI
};

eventSource.onerror = (error) => {
    console.error('SSE Error:', error);
    eventSource.close();
};
```

## 🛠 System Components

### Task Decomposition
The system breaks down complex tasks into subtasks that can be executed by specialized agents:

```python
{
    'website_analysis': ['crawl', 'analyze_content', 'keyword_research'],
    'content_generation': ['keyword_research', 'generate', 'optimize'],
    'content_optimization': ['technical_seo', 'analyze', 'intent']
}
```

### Agent Communication
Agents communicate through a shared context system:
- Task-specific context
- Global parameters
- Intermediate results
- Real-time updates

### Performance Monitoring
The system tracks various metrics:
- Task completion rate
- Average response time
- Error rate
- Agent-specific performance metrics

## 📊 Response Formats

### Task Response
```json
{
    "success": true,
    "data": {
        "task_type": "content_generation",
        "timestamp": "2024-02-04T12:00:00Z",
        "results": {
            "keyword_research": {},
            "content": {},
            "optimization": {}
        }
    },
    "execution_time": 5.2
}
```

### System Status Response
```json
{
    "agents": {
        "content": {"status": "healthy"},
        "seo": {"status": "healthy"},
        "keyword": {"status": "healthy"},
        "crawler": {"status": "healthy"}
    },
    "tasks": 10,
    "active_tasks": 2,
    "system_health": "optimal",
    "last_update": "2024-02-04T12:00:00Z"
}
```

## 🔒 Security

- API key authentication
- Rate limiting
- Input validation
- Secure credential management

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 