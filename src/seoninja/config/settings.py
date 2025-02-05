"""Configuration settings for SEO Ninja."""
from typing import Dict, Any
from pathlib import Path
import os
import yaml

class Settings:
    """Manages configuration settings for SEO Ninja."""
    
    def __init__(self, config_path: str = None):
        """Initialize settings."""
        self.config_path = config_path or os.path.join(
            Path(__file__).parent, "config.yaml"
        )
        self.settings = self._load_settings()
        
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from config file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_settings()
        
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings."""
        return {
            'llm': {
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 2000
            },
            'content': {
                'cache_dir': 'data/cache/content',
                'output_dir': 'data/output/blogs',
                'max_retries': 3
            },
            'seo': {
                'min_word_count': 1500,
                'optimal_keyword_density': 0.02,
                'max_keyword_density': 0.03
            },
            'crawler': {
                'max_pages': 100,
                'timeout': 30,
                'user_agent': 'SEONinja Bot 1.0'
            }
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value."""
        return self.settings.get(key, default)
        
    def update(self, updates: Dict[str, Any]) -> None:
        """Update settings."""
        self.settings.update(updates)

# Create a singleton instance
_settings = Settings()

def get_settings() -> Settings:
    """Get the singleton settings instance."""
    return _settings 