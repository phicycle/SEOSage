"""Dashboard interface for running agents."""
import streamlit as st
import asyncio
from typing import Dict, Any, List
from ..agents.orchestrator.orchestrator import SEOOrchestrator

class AgentRunner:
    """Dashboard interface for running SEO Ninja agents."""
    
    def __init__(self, orchestrator: SEOOrchestrator):
        """Initialize agent runner with existing orchestrator instance."""
        self.orchestrator = orchestrator
        
    def render_dashboard(self):
        """Render the agent runner dashboard."""
        st.title("SEO Ninja Agent Runner")
        
        # Sidebar for task selection
        task = st.sidebar.selectbox(
            "Select Task",
            ["Content Generation", "SEO Audit", "Batch Content Generation"]
        )
        
        if task == "Content Generation":
            self._render_content_generation()
        elif task == "SEO Audit":
            self._render_seo_audit()
        else:
            self._render_batch_generation()
            
    def _render_content_generation(self):
        """Render content generation interface."""
        st.header("Content Generation")
        
        # Input form
        with st.form("content_form"):
            keyword = st.text_input("Target Keyword")
            intent = st.selectbox(
                "Search Intent",
                ["informational", "commercial", "transactional", "navigational"]
            )
            website_url = st.text_input("Website URL (optional)")
            
            submit = st.form_submit_button("Generate Content")
            
        if submit and keyword:
            task = {
                'type': 'content_generation',
                'keyword': keyword,
                'intent': intent,
                'url': website_url if website_url else None
            }
            
            # Run content generation
            with st.spinner("Generating content..."):
                result = asyncio.run(self.orchestrator.execute(task))
                
            if result['success']:
                st.success("Content generated successfully!")
                content = result['data']['content']['content']  # Access nested content
                st.markdown(content)
                st.download_button(
                    "Download Markdown",
                    content,
                    file_name=f"{keyword.replace(' ', '_')}.md"
                )
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
                
    def _render_seo_audit(self):
        """Render SEO audit interface."""
        st.header("SEO Audit")
        
        # Input form
        with st.form("audit_form"):
            url = st.text_input("Website URL")
            submit = st.form_submit_button("Run Audit")
            
        if submit and url:
            task = {
                'type': 'website_analysis',
                'url': url
            }
            
            with st.spinner("Running audit..."):
                result = asyncio.run(self.orchestrator.execute(task))
                
            if result['success']:
                st.success("Audit completed!")
                self._display_audit_results(result['data'])
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
                
    def _render_batch_generation(self):
        """Render batch content generation interface."""
        st.header("Batch Content Generation")
        
        # Input form
        with st.form("batch_form"):
            keywords_text = st.text_area(
                "Enter keywords (one per line)"
            )
            intent = st.selectbox(
                "Search Intent",
                ["informational", "commercial", "transactional", "navigational"]
            )
            website_url = st.text_input("Website URL (optional)")
            
            submit = st.form_submit_button("Generate Batch Content")
            
        if submit and keywords_text:
            keywords = [k.strip() for k in keywords_text.split('\n') if k.strip()]
            
            tasks = [
                {
                    'type': 'content_generation',
                    'keyword': keyword,
                    'intent': intent,
                    'url': website_url if website_url else None
                }
                for keyword in keywords
            ]
            
            with st.spinner("Generating batch content..."):
                results = []
                for task in tasks:
                    result = asyncio.run(self.orchestrator.execute(task))
                    results.append(result)
                
            self._display_batch_results(results)
            
    def _display_audit_results(self, results: Dict[str, Any]):
        """Display SEO audit results."""
        st.subheader("Audit Results")
        
        # Display crawl results
        if 'structure' in results:
            with st.expander("Site Structure"):
                st.json(results['structure'])
                
        # Display content analysis
        if 'content' in results:
            with st.expander("Content Analysis"):
                st.json(results['content'])
                
        # Display keyword analysis
        if 'keywords' in results:
            with st.expander("Keyword Analysis"):
                st.json(results['keywords'])
                    
    def _display_batch_results(self, results: List[Dict[str, Any]]):
        """Display batch generation results."""
        st.subheader("Batch Results")
        
        for result in results:
            if result['success']:
                keyword = result['data']['content'].get('keyword', 'Unknown')
                content = result['data']['content'].get('content', '')
                
                with st.expander(f"Content for: {keyword}"):
                    st.markdown(content)
                    st.download_button(
                        "Download Markdown",
                        content,
                        file_name=f"{keyword.replace(' ', '_')}.md"
                    )
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
                
    def cleanup(self):
        """Cleanup orchestrator resources."""
        asyncio.run(self.orchestrator.shutdown()) 