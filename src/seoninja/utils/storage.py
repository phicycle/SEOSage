"""Persistent storage system for agent memory and state."""
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import logging
import sqlite3

class PersistentStorage:
    """Manages persistent storage for agent memory and state."""
    
    def __init__(self, base_dir: str = 'data/storage'):
        """Initialize storage system."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup SQLite database
        self.db_path = self.base_dir / 'agent_storage.db'
        self._ensure_db_exists()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _ensure_db_exists(self) -> None:
        """Ensure database and tables exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_state (
                        agent_name TEXT PRIMARY KEY,
                        state TEXT NOT NULL,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise
            
    async def save(self, key: str, data: Any) -> None:
        """Save data to storage."""
        try:
            file_path = self.base_dir / f"{key}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            
    async def load(self, key: str) -> Optional[Any]:
        """Load data from storage."""
        try:
            file_path = self.base_dir / f"{key}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
        return None
        
    def save_memory(self, agent_name: str, memory_type: str, content: Dict[str, Any]) -> None:
        """Save agent memory to persistent storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO agent_memory (agent_name, memory_type, content) VALUES (?, ?, ?)",
                    (agent_name, memory_type, json.dumps(content))
                )
        except Exception as e:
            self.logger.error(f"Error saving memory: {str(e)}")
            
    def get_memories(self, agent_name: str, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve agent memories from storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if memory_type:
                    cursor = conn.execute(
                        """SELECT content, timestamp 
                           FROM agent_memory 
                           WHERE agent_name = ? AND memory_type = ? 
                           ORDER BY timestamp DESC""",
                        (agent_name, memory_type)
                    )
                else:
                    cursor = conn.execute(
                        """SELECT content, timestamp 
                           FROM agent_memory 
                           WHERE agent_name = ? 
                           ORDER BY timestamp DESC""",
                        (agent_name,)
                    )
                
                return [
                    {
                        'content': json.loads(content),
                        'timestamp': timestamp
                    }
                    for content, timestamp in cursor.fetchall()
                ]
        except Exception as e:
            self.logger.error(f"Error getting memories: {str(e)}")
            return []
            
    async def save_state(self, key: str, state: Dict[str, Any]) -> None:
        """Save state asynchronously."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO agent_state (agent_name, state)
                       VALUES (?, ?)""",
                    (key, json.dumps(state))
                )
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            
    def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state synchronously."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT state FROM agent_state WHERE agent_name = ?",
                    (key,)
                )
                result = cursor.fetchone()
                return json.loads(result[0]) if result else None
        except Exception as e:
            self.logger.error(f"Error getting state: {str(e)}")
            return None
            
    def save_metric(self, agent: str, metric: str, value: Any) -> None:
        """Save a metric value."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO metrics (agent_name, metric_name, value) VALUES (?, ?, ?)",
                    (agent, metric, value)
                )
        except Exception as e:
            self.logger.error(f"Error saving metric: {str(e)}")
            
    def get_metrics(self, agent: str, metric: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get metrics for an agent."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if metric:
                    cursor = conn.execute(
                        """SELECT metric_name, value, timestamp 
                           FROM metrics 
                           WHERE agent_name = ? AND metric_name = ? 
                           ORDER BY timestamp DESC""",
                        (agent, metric)
                    )
                else:
                    cursor = conn.execute(
                        """SELECT metric_name, value, timestamp 
                           FROM metrics 
                           WHERE agent_name = ? 
                           ORDER BY timestamp DESC""",
                        (agent,)
                    )
                
                return [
                    {
                        'metric': name,
                        'value': value,
                        'timestamp': timestamp
                    }
                    for name, value, timestamp in cursor.fetchall()
                ]
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return [] 