"""Caching utility for SEO Ninja."""
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import logging

class Cache:
    """Cache manager for storing and retrieving analysis results."""
    
    def __init__(self, namespace: str, cache_duration: int = 30 * 24 * 60 * 60):
        """
        Initialize cache manager.
        
        Args:
            namespace: Cache namespace (e.g., 'crawler', 'keyword_research')
            cache_duration: Cache duration in seconds (default: 30 days)
        """
        self.namespace = namespace
        self.cache_duration = cache_duration
        self.cache_dir = Path("data/cache") / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"cache.{namespace}")
        
    def _get_cache_path(self, key: str, analysis_type: str) -> Path:
        """Get path for specific cache file."""
        # Create a unique filename from the key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}_{analysis_type}.pkl"
        
    def load(self, key: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """
        Load cached data if available and fresh.
        
        Args:
            key: Cache key (e.g., domain name)
            analysis_type: Type of analysis (e.g., 'crawl', 'keywords')
            
        Returns:
            Cached data if available and fresh, None otherwise
        """
        cache_path = self._get_cache_path(key, analysis_type)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                    # Check if cache is fresh
                    if datetime.now().timestamp() - cached_data.get('timestamp', 0) < self.cache_duration:
                        self.logger.debug(
                            f"Using cached {analysis_type} data for {key} "
                            f"(age: {(datetime.now().timestamp() - cached_data.get('timestamp', 0)) / (24 * 60 * 60):.1f} days)"
                        )
                        return cached_data.get('data')
                    else:
                        self.logger.debug(f"Cache expired for {analysis_type} data of {key}")
                        
            except Exception as e:
                self.logger.warning(f"Error loading {analysis_type} cache for {key}: {str(e)}")
                
        return None
        
    def save(self, key: str, analysis_type: str, data: Any) -> None:
        """
        Save data to cache.
        
        Args:
            key: Cache key (e.g., domain name)
            analysis_type: Type of analysis (e.g., 'crawl', 'keywords')
            data: Data to cache
        """
        try:
            cache_path = self._get_cache_path(key, analysis_type)
            
            cache_data = {
                'timestamp': datetime.now().timestamp(),
                'data': data
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            self.logger.debug(f"Saved {analysis_type} cache for {key}")
            
        except Exception as e:
            self.logger.error(f"Error saving {analysis_type} cache for {key}: {str(e)}")
            
    def clear(self, key: Optional[str] = None, analysis_type: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            key: Optional key to clear specific entries
            analysis_type: Optional analysis type to clear specific entries
        """
        try:
            if key and analysis_type:
                # Clear specific cache file
                cache_path = self._get_cache_path(key, analysis_type)
                if cache_path.exists():
                    cache_path.unlink()
                    
            elif key:
                # Clear all cache files for key
                safe_key = hashlib.md5(key.encode()).hexdigest()
                for cache_file in self.cache_dir.glob(f"{safe_key}_*.pkl"):
                    cache_file.unlink()
                    
            elif analysis_type:
                # Clear all cache files for analysis type
                for cache_file in self.cache_dir.glob(f"*_{analysis_type}.pkl"):
                    cache_file.unlink()
                    
            else:
                # Clear all cache files in namespace
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    
            self.logger.info(f"Cleared cache: key={key}, type={analysis_type}")
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'total_files': 0,
            'total_size': 0,
            'oldest_file': None,
            'newest_file': None,
            'analysis_types': {}
        }
        
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                stats['total_files'] += 1
                stats['total_size'] += cache_file.stat().st_size
                
                # Track analysis types
                analysis_type = cache_file.stem.split('_')[-1]
                if analysis_type not in stats['analysis_types']:
                    stats['analysis_types'][analysis_type] = 0
                stats['analysis_types'][analysis_type] += 1
                
                # Track file ages
                mtime = cache_file.stat().st_mtime
                if not stats['oldest_file'] or mtime < stats['oldest_file']:
                    stats['oldest_file'] = mtime
                if not stats['newest_file'] or mtime > stats['newest_file']:
                    stats['newest_file'] = mtime
                    
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            
        return stats 