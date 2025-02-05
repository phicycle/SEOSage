"""SEO Ninja - An agent-based SEO optimization toolkit."""
from .agents.orchestrator.orchestrator import SEOOrchestrator
from .config.settings import get_settings

__version__ = "0.1.0"
__all__ = ["SEOOrchestrator", "get_settings"] 