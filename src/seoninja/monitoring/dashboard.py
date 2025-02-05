"""Monitoring dashboard for agent activities."""
import dash
from dash import html, dcc, Dash
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from ..utils.storage import PersistentStorage
from ..config.settings import get_settings
import plotly.express as px

class Dashboard:
    """Dashboard for monitoring agent activities."""
    
    def __init__(self, storage: Optional[PersistentStorage] = None):
        """Initialize dashboard with optional storage."""
        print("Initializing Dashboard with storage:", storage)  # Debug print
        try:
            self.storage = storage if storage is not None else PersistentStorage()
            print("Storage initialized successfully")
            
            # Initialize Dash app
            print("Initializing Dash app")
            self.app = Dash(
                name=__name__,
                external_stylesheets=[
                    'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
                ],
                suppress_callback_exceptions=True,
                title='SEO Ninja Dashboard'
            )
            print("Dash app initialized successfully")
            
            self._setup_layout()
            self._setup_callbacks()
            print("Dashboard setup complete")
        except Exception as e:
            print(f"Error during Dashboard initialization: {str(e)}")
            raise
        
    def run(self, host: str = 'localhost', port: int = 8050, debug: bool = False):
        """Run the dashboard server."""
        self.app.run_server(host=host, port=port, debug=debug)

    def _setup_layout(self) -> None:
        """Set up dashboard layout."""
        self.app.layout = html.Div([
            # Navigation Bar
            html.Nav([
                html.H1('SEO Ninja Monitoring Dashboard', className='navbar-brand text-white'),
                html.Div([
                    html.Span('Last Updated: ', className='text-white'),
                    html.Span(id='last-update-time', className='text-white')
                ])
            ], className='navbar navbar-dark bg-dark mb-4 p-3'),
            
            # Main Content Container
            html.Div([
                # Date Range Selector
                html.Div([
                    html.Label('Select Date Range:'),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=(datetime.now() - timedelta(days=7)).date(),
                        end_date=datetime.now().date(),
                        className='mb-4'
                    )
                ], className='mb-4'),
                
                # System Health Indicators
                html.Div([
                    html.H3('System Health', className='mb-3'),
                    html.Div([
                        html.Div([
                            html.Div(id='system-health-indicators', className='d-flex justify-content-between')
                        ], className='card-body')
                    ], className='card mb-4')
                ]),
                
                # Performance Metrics
                html.Div([
                    html.H3('Performance Metrics', className='mb-3'),
                    html.Div([
                        # Task Success/Failure Metrics
                        html.Div([
                            dcc.Graph(id='task-success-rate'),
                        ], className='col-md-6'),
                        # Execution Time Metrics
                        html.Div([
                            dcc.Graph(id='execution-time-graph'),
                        ], className='col-md-6'),
                    ], className='row mb-4'),
                ]),
                
                # SEO Metrics
                html.Div([
                    html.H3('SEO Metrics', className='mb-3'),
                    html.Div([
                        html.Div([
                            dcc.Graph(id='keyword-rankings'),
                        ], className='col-md-6'),
                        html.Div([
                            dcc.Graph(id='organic-traffic'),
                        ], className='col-md-6'),
                    ], className='row mb-4')
                ]),
                
                # Resource Utilization
                html.Div([
                    html.H3('Resource Utilization', className='mb-3'),
                    html.Div([
                        html.Div([
                            dcc.Graph(id='api-usage'),
                        ], className='col-md-6'),
                        html.Div([
                            dcc.Graph(id='memory-usage'),
                        ], className='col-md-6'),
                    ], className='row mb-4')
                ]),
                
                # Agent Activities and States
                html.Div([
                    html.Div([
                        html.H3('Agent Activities', className='mb-3'),
                        html.Div(id='recent-activities', className='card p-3')
                    ], className='col-md-6'),
                    html.Div([
                        html.H3('Agent States', className='mb-3'),
                        html.Div(id='agent-states', className='card p-3')
                    ], className='col-md-6'),
                ], className='row mb-4'),
                
                # Rate Limits and API Status
                html.Div([
                    html.H3('API Status & Rate Limits', className='mb-3'),
                    html.Div(id='rate-limits', className='card p-3')
                ], className='mb-4'),
                
                dcc.Interval(
                    id='interval-component',
                    interval=30*1000,  # 30 seconds
                    n_intervals=0
                )
            ], className='container-fluid')
        ])
        
    def _setup_callbacks(self) -> None:
        """Set up dashboard callbacks."""
        @self.app.callback(
            [Output('last-update-time', 'children'),
             Output('system-health-indicators', 'children'),
             Output('task-success-rate', 'figure'),
             Output('execution-time-graph', 'figure'),
             Output('keyword-rankings', 'figure'),
             Output('organic-traffic', 'figure'),
             Output('api-usage', 'figure'),
             Output('memory-usage', 'figure'),
             Output('recent-activities', 'children'),
             Output('agent-states', 'children'),
             Output('rate-limits', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_dashboard(_, start_date, end_date):
            """Update all dashboard components."""
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Generate all components
            health_indicators = self._generate_health_indicators()
            success_rate = self._generate_success_rate_graph(start_date, end_date)
            execution_time = self._generate_execution_time_graph(start_date, end_date)
            rankings = self._generate_keyword_rankings(start_date, end_date)
            traffic = self._generate_organic_traffic(start_date, end_date)
            api_usage = self._generate_api_usage_graph(start_date, end_date)
            memory_usage = self._generate_memory_usage_graph()
            activities = self._generate_activities_list()
            states = self._generate_states_display()
            limits = self._generate_rate_limits_display()
            
            return (
                current_time, health_indicators, success_rate, execution_time,
                rankings, traffic, api_usage, memory_usage, activities,
                states, limits
            )

    def _generate_health_indicators(self) -> List[html.Div]:
        """Generate system health status indicators."""
        indicators = []
        services = ['Orchestrator', 'Content Generator', 'Keyword Research', 'Crawler']
        
        for service in services:
            # Get service status from storage
            status = self.storage.get_state(service.lower().replace(' ', '_')) or {'healthy': False}
            status_color = 'success' if status.get('healthy', False) else 'danger'
            
            indicators.append(html.Div([
                html.H5(service),
                html.Div([
                    html.Div(className=f'bg-{status_color} rounded-circle', 
                            style={'width': '20px', 'height': '20px'}),
                    html.Small(f"Last check: {status.get('last_check', 'N/A')}")
                ], className='d-flex align-items-center gap-2')
            ], className='text-center'))
            
        return indicators

    def _generate_success_rate_graph(self, start_date: str, end_date: str) -> go.Figure:
        """Generate task success rate graph."""
        metrics = []
        for agent_name in ['orchestrator', 'content', 'keyword', 'crawler']:
            success_metrics = self.storage.get_metrics(
                agent_name,
                'task_success'
            )
            metrics.extend([{
                'agent': agent_name,
                'time': m['timestamp'],
                'success': m['value']
            } for m in success_metrics])
        
        if not metrics:
            return go.Figure()
            
        df = pd.DataFrame(metrics)
        df['date'] = pd.to_datetime(df['time'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        success_rates = df.groupby('agent')['success'].mean().reset_index()
        
        fig = px.bar(success_rates, x='agent', y='success',
                    title='Task Success Rate by Agent',
                    labels={'success': 'Success Rate', 'agent': 'Agent'},
                    color='agent')
        
        return fig

    def _generate_keyword_rankings(self, start_date: str, end_date: str) -> go.Figure:
        """Generate keyword rankings graph."""
        rankings = self.storage.get_metrics('keywords', 'rankings')
        if not rankings:
            return go.Figure()
            
        df = pd.DataFrame(rankings)
        df['date'] = pd.to_datetime(df['timestamp'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        fig = px.line(df, x='date', y='value',
                     title='Keyword Rankings Over Time',
                     labels={'value': 'Average Position', 'date': 'Date'})
        
        return fig

    def _generate_organic_traffic(self, start_date: str, end_date: str) -> go.Figure:
        """Generate organic traffic graph."""
        traffic = self.storage.get_metrics('analytics', 'organic_traffic')
        if not traffic:
            return go.Figure()
            
        df = pd.DataFrame(traffic)
        df['date'] = pd.to_datetime(df['timestamp'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        fig = px.line(df, x='date', y='value',
                     title='Organic Traffic Over Time',
                     labels={'value': 'Visitors', 'date': 'Date'})
        
        return fig

    def _generate_api_usage_graph(self, start_date: str, end_date: str) -> go.Figure:
        """Generate API usage graph."""
        services = ['openai', 'moz', 'semrush', 'google']
        usage_data = []
        
        for service in services:
            metrics = self.storage.get_metrics(service, 'requests')
            for metric in metrics:
                usage_data.append({
                    'service': service,
                    'time': metric['timestamp'],
                    'requests': metric['value']
                })
        
        if not usage_data:
            return go.Figure()
            
        df = pd.DataFrame(usage_data)
        df['date'] = pd.to_datetime(df['time'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        fig = px.line(df, x='date', y='requests', color='service',
                     title='API Usage by Service',
                     labels={'requests': 'Number of Requests', 'date': 'Date'})
        
        return fig

    def _generate_memory_usage_graph(self) -> go.Figure:
        """Generate memory usage graph."""
        memory_metrics = self.storage.get_metrics('system', 'memory_usage')
        if not memory_metrics:
            return go.Figure()
            
        df = pd.DataFrame(memory_metrics)
        df['time'] = pd.to_datetime(df['timestamp'])
        
        fig = px.line(df, x='time', y='value',
                     title='Memory Usage Over Time',
                     labels={'value': 'Memory Usage (MB)', 'time': 'Time'})
        
        return fig

    def _generate_execution_time_graph(self, start_date: str, end_date: str) -> go.Figure:
        """Generate execution time graph."""
        # Get metrics for all agents
        metrics = []
        for agent_name in ['orchestrator', 'content']:
            agent_metrics = self.storage.get_metrics(
                agent_name,
                'task_execution_time'
            )
            for metric in agent_metrics:
                metrics.append({
                    'agent': agent_name,
                    'time': metric['timestamp'],
                    'value': metric['value']
                })
                
        if not metrics:
            return go.Figure()
            
        df = pd.DataFrame(metrics)
        df['date'] = pd.to_datetime(df['time'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        return {
            'data': [
                go.Scatter(
                    x=df[df['agent'] == agent]['date'],
                    y=df[df['agent'] == agent]['value'],
                    name=agent,
                    mode='lines+markers'
                )
                for agent in df['agent'].unique()
            ],
            'layout': {
                'title': 'Task Execution Times',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Execution Time (s)'}
            }
        }
        
    def _generate_activities_list(self) -> html.Div:
        """Generate recent activities list."""
        activities = []
        for agent_name in ['orchestrator', 'content']:
            memories = self.storage.get_memories(agent_name)
            for memory in memories[:5]:  # Last 5 memories
                activities.append(
                    html.Div([
                        html.Strong(f"{agent_name}: "),
                        html.Span(str(memory['content'])),
                        html.Br(),
                        html.Small(memory['timestamp'])
                    ])
                )
        
        return html.Div(activities)
        
    def _generate_states_display(self) -> html.Div:
        """Generate agent states display."""
        states = []
        for agent_name in ['orchestrator', 'content']:
            state = self.storage.get_state(agent_name)
            if state:
                states.append(
                    html.Div([
                        html.Strong(f"{agent_name} State:"),
                        html.Pre(str(state))
                    ])
                )
        
        return html.Div(states)
        
    def _generate_rate_limits_display(self) -> html.Div:
        """Generate rate limits display."""
        services = ['openai', 'moz', 'semrush']
        limits = []
        
        for service in services:
            metrics = self.storage.get_metrics(service, 'requests')
            if metrics:
                recent_requests = len([
                    m for m in metrics
                    if datetime.fromisoformat(m['timestamp']) > datetime.now() - timedelta(minutes=1)
                ])
                limits.append(
                    html.Div([
                        html.Strong(f"{service}: "),
                        html.Span(f"{recent_requests} requests in last minute")
                    ])
                )
        
        return html.Div(limits) 