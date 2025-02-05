"""Run the SEO Ninja dashboard."""
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from .agents.orchestrator.orchestrator import SEOOrchestrator
from .dashboard.agent_runner import AgentRunner
from .utils.storage import PersistentStorage
from .utils.gsc_auth import get_gsc_credentials

def run_dashboard():
    # Load environment variables
    load_dotenv()
    
    # Get required credentials and configuration
    gsc_credentials = get_gsc_credentials(os.getenv('GSC_CLIENT_SECRETS_PATH', 'client_secrets.json'))
    moz_token = os.getenv('MOZ_API_TOKEN')
    domain = os.getenv('TARGET_DOMAIN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # Set page config
    st.set_page_config(
        page_title="SEO Ninja Dashboard",
        page_icon="🥷",
        layout="wide"
    )

    # Create new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Initialize storage and orchestrator with proper config
    storage = PersistentStorage()
    orchestrator = SEOOrchestrator(
        storage=storage,
        gsc_credentials=gsc_credentials,
        moz_token=moz_token,
        target_domain=domain,
        openai_api_key=openai_api_key
    )
    
    # Initialize async components
    loop.run_until_complete(orchestrator.run())
    
    # Create and run dashboard
    runner = AgentRunner(orchestrator)
    try:
        runner.render_dashboard()
    finally:
        loop.run_until_complete(orchestrator.shutdown())
        loop.close() 