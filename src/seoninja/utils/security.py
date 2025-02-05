"""Security and rate limiting utilities."""
import time
from typing import Dict, Any, Optional
import os
from datetime import datetime, timedelta
import json
from cryptography.fernet import Fernet
from ..config.settings import get_settings

class APIKeyManager:
    """Manages API keys securely."""
    
    def __init__(self):
        """Initialize API key manager."""
        self.key_file = os.path.join(
            get_settings().get('content.cache_dir', 'data/cache'),
            '.api_keys'
        )
        self._ensure_key_file()
        self._load_encryption_key()
        
    def _ensure_key_file(self) -> None:
        """Ensure API key file exists."""
        os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
        if not os.path.exists(self.key_file):
            with open(self.key_file, 'wb') as f:
                f.write(b'{}')
                
    def _load_encryption_key(self) -> None:
        """Load or generate encryption key."""
        key_path = os.path.join(os.path.dirname(self.key_file), '.encryption_key')
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(self.key)
        self.cipher = Fernet(self.key)
        
    def set_api_key(self, service: str, key: str) -> None:
        """Securely store API key."""
        encrypted_key = self.cipher.encrypt(key.encode())
        with open(self.key_file, 'rb') as f:
            keys = json.loads(f.read())
        keys[service] = encrypted_key.decode()
        with open(self.key_file, 'wb') as f:
            f.write(json.dumps(keys).encode())
            
    def get_api_key(self, service: str) -> Optional[str]:
        """Retrieve API key."""
        try:
            with open(self.key_file, 'rb') as f:
                keys = json.loads(f.read())
            if service in keys:
                return self.cipher.decrypt(keys[service].encode()).decode()
        except Exception:
            return None
        return None

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.limits: Dict[str, Dict[str, Any]] = {}
        self.requests: Dict[str, list] = {}
        
    def add_limit(self, service: str, requests: int, period: int) -> None:
        """Add rate limit for a service."""
        self.limits[service] = {
            'requests': requests,
            'period': period
        }
        self.requests[service] = []
        
    def can_request(self, service: str) -> bool:
        """Check if request is allowed."""
        if service not in self.limits:
            return True
            
        now = time.time()
        limit = self.limits[service]
        
        # Clean old requests
        self.requests[service] = [
            req_time for req_time in self.requests[service]
            if now - req_time <= limit['period']
        ]
        
        return len(self.requests[service]) < limit['requests']
        
    def add_request(self, service: str) -> None:
        """Record a request."""
        if service in self.requests:
            self.requests[service].append(time.time())
            
class InputValidator:
    """Input validation and sanitization."""
    
    @staticmethod
    def validate_keyword_data(data: Dict[str, Any]) -> Optional[str]:
        """Validate keyword data."""
        required = ['keyword', 'intent']
        if missing := [f for f in required if f not in data]:
            return f"Missing required fields: {', '.join(missing)}"
            
        if not isinstance(data['keyword'], str):
            return "Keyword must be a string"
            
        if data['intent'] not in ['informational', 'commercial', 'transactional', 'navigational']:
            return "Invalid intent value"
            
        return None
        
    @staticmethod
    def sanitize_content(content: str) -> str:
        """Sanitize content input."""
        # Remove potential XSS/injection patterns
        dangerous_patterns = [
            '<script>',
            'javascript:',
            'data:',
            'vbscript:',
            'onload=',
            'onerror='
        ]
        
        sanitized = content
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
            
        return sanitized
        
    @staticmethod
    def validate_url(url: str) -> Optional[str]:
        """Validate URL format."""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            
        if not url_pattern.match(url):
            return "Invalid URL format"
        return None 