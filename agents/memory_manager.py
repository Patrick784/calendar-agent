"""
Memory Manager

Manages three memory tiers:
1. Short-term scratchpad: summarizes the current conversation and plan
2. Mid-term task board: stored in a shared document or database  
3. Long-term vector store: (pgvector or Chroma) storing past tasks, feedback and preferences

The orchestrator retrieves relevant context from this store.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import os
import logging

# Optional dependencies with graceful fallbacks
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using in-memory cache for scratchpad")

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logging.warning("PostgreSQL/pgvector not available - using in-memory storage for task board")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available - using simple in-memory vector store")

class MemoryManager:
    """
    Manages multi-tier memory system for the calendar agent.
    
    Tiers:
    1. Scratchpad (Redis/in-memory): Current conversation context
    2. Task Board (PostgreSQL): Active and completed tasks
    3. Long-term Memory (pgvector/Chroma): Historical interactions and preferences
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("memory_manager")
        
        # Feature flags for memory components
        self.vector_memory_enabled = os.getenv("VECTOR_MEMORY_ENABLED", "true").lower() == "true"
        self.persistent_memory_enabled = os.getenv("PERSISTENT_MEMORY_ENABLED", "true").lower() == "true"
        
        # Initialize storage backends
        self._scratchpad_store = None
        self._task_board_store = None  
        self._vector_store = None
        
        # In-memory fallbacks
        self._scratchpad_memory: Dict[str, Any] = {}
        self._task_board_memory: List[Dict[str, Any]] = []
        
        # Determine which backends to use
        self.use_redis = REDIS_AVAILABLE and self.persistent_memory_enabled and self.config.get("redis_url")
        self.use_postgres = POSTGRES_AVAILABLE and self.persistent_memory_enabled and self.config.get("postgres_url")
        self.use_vector_store = (CHROMA_AVAILABLE or POSTGRES_AVAILABLE) and self.vector_memory_enabled
        
        # Log memory configuration
        self.logger.info(f"Memory system configuration:")
        self.logger.info(f"  Redis scratchpad: {'enabled' if self.use_redis else 'disabled (using in-memory)'}")
        self.logger.info(f"  PostgreSQL task board: {'enabled' if self.use_postgres else 'disabled (using in-memory)'}")
        self.logger.info(f"  Vector store: {'enabled' if self.use_vector_store else 'disabled (using simple memory)'}")
        
        # Initialize connections only if backends are to be used
        if self.use_redis or not self.persistent_memory_enabled:
            self._setup_scratchpad()
        if self.use_postgres or not self.persistent_memory_enabled:
            self._setup_task_board()
        if self.use_vector_store or not self.vector_memory_enabled:
            self._setup_vector_store()
    
    def _setup_scratchpad(self):
        """Initialize short-term memory scratchpad (Redis preferred, in-memory fallback)"""
        
        if self.use_redis:
            try:
                self._scratchpad_store = redis.from_url(
                    self.config["redis_url"],
                    decode_responses=True
                )
                # Test connection
                self._scratchpad_store.ping()
                self.logger.info("Connected to Redis for scratchpad storage")
                return
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {str(e)}, falling back to in-memory")
                self.use_redis = False
        
        self.logger.info("Using in-memory scratchpad storage")
        self._scratchpad_store = None
    
    def _setup_task_board(self):
        """Initialize mid-term task board storage (PostgreSQL preferred)"""
        
        if self.use_postgres:
            try:
                self._task_board_store = psycopg2.connect(self.config["postgres_url"])
                register_vector(self._task_board_store)
                
                # Create tables if they don't exist
                self._create_task_board_tables()
                self.logger.info("Connected to PostgreSQL for task board storage")
                return
            except Exception as e:
                self.logger.warning(f"PostgreSQL connection failed: {str(e)}, falling back to in-memory")
                self.use_postgres = False
        
        self.logger.info("Using in-memory task board storage")
        self._task_board_store = None
    
    def _setup_vector_store(self):
        """Initialize long-term vector memory (pgvector or Chroma)"""
        
        if not self.use_vector_store:
            self.logger.info("Vector store disabled by configuration")
            self._vector_store = None
            return
        
        # Try pgvector first (if PostgreSQL is available)
        if self.use_postgres and self._task_board_store and self.config.get("use_pgvector", True):
            try:
                self._create_vector_tables()
                self._vector_store = "pgvector"
                self.logger.info("Using pgvector for long-term memory")
                return
            except Exception as e:
                self.logger.warning(f"pgvector setup failed: {str(e)}")
        
        # Fallback to Chroma
        if CHROMA_AVAILABLE:
            try:
                chroma_path = self.config.get("chroma_path", "./chroma_db")
                client = chromadb.PersistentClient(path=chroma_path)
                
                # Create collections with better error handling
                try:
                    self._interactions_collection = client.get_or_create_collection(
                        name="interactions",
                        metadata={"description": "Historical user interactions"}
                    )
                    self._preferences_collection = client.get_or_create_collection(
                        name="preferences", 
                        metadata={"description": "User preferences and patterns"}
                    )
                    
                    self._vector_store = "chroma"
                    self.logger.info("Using Chroma for long-term memory")
                    return
                except Exception as collection_error:
                    self.logger.warning(f"Chroma collection creation failed: {str(collection_error)}")
                    raise collection_error
                    
            except Exception as e:
                self.logger.warning(f"Chroma setup failed: {str(e)}")
        
        # Fallback to simple in-memory store - this should always work
        self.logger.info("Using simple in-memory store for long-term memory")
        self._vector_store = "memory"
        self._interactions_memory = []
        self._preferences_memory = []
    
    def _create_task_board_tables(self):
        """Create PostgreSQL tables for task board"""
        
        with self._task_board_store.cursor() as cur:
            # Tasks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id SERIAL PRIMARY KEY,
                    task_id VARCHAR(255) UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    status VARCHAR(50) DEFAULT 'pending',
                    priority INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    due_date TIMESTAMP,
                    completed_at TIMESTAMP,
                    metadata JSONB,
                    user_id VARCHAR(255)
                )
            """)
            
            # Interactions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id SERIAL PRIMARY KEY,
                    interaction_id VARCHAR(255) UNIQUE NOT NULL,
                    user_request TEXT NOT NULL,
                    intent VARCHAR(100),
                    success BOOLEAN,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_data JSONB,
                    plan_data JSONB,
                    user_id VARCHAR(255)
                )
            """)
            
            # Evaluations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id SERIAL PRIMARY KEY,
                    plan_id VARCHAR(255) NOT NULL,
                    success BOOLEAN,
                    steps_planned INTEGER,
                    agents_used TEXT[],
                    tools_used TEXT[],
                    insights TEXT[],
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self._task_board_store.commit()
    
    def _create_vector_tables(self):
        """Create pgvector tables for long-term memory"""
        
        with self._task_board_store.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Embeddings table for semantic search
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    id SERIAL PRIMARY KEY,
                    content_hash VARCHAR(64) UNIQUE NOT NULL,
                    content_type VARCHAR(50) NOT NULL,
                    content_text TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id VARCHAR(255)
                )
            """)
            
            # Create index for similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS memory_embeddings_embedding_idx 
                ON memory_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            self._task_board_store.commit()
    
    async def update_scratchpad(self, data: Dict[str, Any], expire_seconds: int = 3600):
        """Update short-term scratchpad with current conversation context"""
        
        key = "current_session"
        serialized_data = json.dumps(data, default=str)
        
        if self._scratchpad_store:
            try:
                self._scratchpad_store.setex(key, expire_seconds, serialized_data)
                return
            except Exception as e:
                self.logger.error(f"Redis scratchpad update failed: {str(e)}")
        
        # Fallback to in-memory
        self._scratchpad_memory[key] = {
            "data": data,
            "expires_at": datetime.utcnow() + timedelta(seconds=expire_seconds)
        }
    
    async def get_scratchpad(self) -> Optional[Dict[str, Any]]:
        """Retrieve current scratchpad content"""
        
        key = "current_session"
        
        if self._scratchpad_store:
            try:
                data = self._scratchpad_store.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                self.logger.error(f"Redis scratchpad retrieval failed: {str(e)}")
        
        # Fallback to in-memory
        if key in self._scratchpad_memory:
            entry = self._scratchpad_memory[key]
            if datetime.utcnow() < entry["expires_at"]:
                return entry["data"]
            else:
                del self._scratchpad_memory[key]
        
        return None
    
    async def store_task(self, task_data: Dict[str, Any]) -> str:
        """Store a task in the mid-term task board"""
        
        task_id = task_data.get("task_id") or self._generate_task_id(task_data)
        
        if self._task_board_store:
            try:
                with self._task_board_store.cursor() as cur:
                    cur.execute("""
                        INSERT INTO tasks (
                            task_id, title, description, status, priority,
                            due_date, metadata, user_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (task_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            status = EXCLUDED.status,
                            priority = EXCLUDED.priority,
                            due_date = EXCLUDED.due_date,
                            metadata = EXCLUDED.metadata,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        task_id,
                        task_data.get("title", ""),
                        task_data.get("description", ""),
                        task_data.get("status", "pending"),
                        task_data.get("priority", 1),
                        task_data.get("due_date"),
                        json.dumps(task_data.get("metadata", {})),
                        task_data.get("user_id", "default")
                    ))
                    self._task_board_store.commit()
                    return task_id
            except Exception as e:
                self.logger.error(f"Task storage failed: {str(e)}")
        
        # Fallback to in-memory
        task_data["task_id"] = task_id
        task_data["created_at"] = datetime.utcnow().isoformat()
        
        # Remove existing task with same ID
        self._task_board_memory = [t for t in self._task_board_memory if t.get("task_id") != task_id]
        self._task_board_memory.append(task_data)
        
        return task_id
    
    async def get_tasks(self, status: Optional[str] = None, user_id: str = "default") -> List[Dict[str, Any]]:
        """Retrieve tasks from the task board"""
        
        if self._task_board_store:
            try:
                with self._task_board_store.cursor() as cur:
                    if status:
                        cur.execute("""
                            SELECT task_id, title, description, status, priority,
                                   created_at, due_date, completed_at, metadata
                            FROM tasks 
                            WHERE user_id = %s AND status = %s
                            ORDER BY created_at DESC
                        """, (user_id, status))
                    else:
                        cur.execute("""
                            SELECT task_id, title, description, status, priority,
                                   created_at, due_date, completed_at, metadata
                            FROM tasks 
                            WHERE user_id = %s
                            ORDER BY created_at DESC
                        """, (user_id,))
                    
                    rows = cur.fetchall()
                    tasks = []
                    for row in rows:
                        task = {
                            "task_id": row[0],
                            "title": row[1],
                            "description": row[2],
                            "status": row[3],
                            "priority": row[4],
                            "created_at": row[5].isoformat() if row[5] else None,
                            "due_date": row[6].isoformat() if row[6] else None,
                            "completed_at": row[7].isoformat() if row[7] else None,
                            "metadata": row[8] or {}
                        }
                        tasks.append(task)
                    return tasks
            except Exception as e:
                self.logger.error(f"Task retrieval failed: {str(e)}")
        
        # Fallback to in-memory
        tasks = self._task_board_memory.copy()
        if status:
            tasks = [t for t in tasks if t.get("status") == status]
        
        return tasks
    
    async def store_interaction(self, interaction_data: Dict[str, Any]):
        """Store completed interaction in both task board and long-term memory"""
        
        interaction_id = self._generate_interaction_id(interaction_data)
        
        # Store in task board
        if self._task_board_store:
            try:
                with self._task_board_store.cursor() as cur:
                    cur.execute("""
                        INSERT INTO interactions (
                            interaction_id, user_request, intent, success,
                            response_data, plan_data, user_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        interaction_id,
                        interaction_data.get("user_request", ""),
                        interaction_data.get("intent", ""),
                        interaction_data.get("success", False),
                        json.dumps(interaction_data.get("result_data", {})),
                        json.dumps(interaction_data.get("plan_data", {})),
                        interaction_data.get("user_id", "default")
                    ))
                    self._task_board_store.commit()
            except Exception as e:
                self.logger.error(f"Interaction storage failed: {str(e)}")
        
        # Store in long-term vector memory for semantic search
        await self._store_in_vector_memory("interaction", interaction_data)
    
    async def store_evaluation(self, evaluation_data: Dict[str, Any]):
        """Store self-evaluation data for learning"""
        
        if self._task_board_store:
            try:
                with self._task_board_store.cursor() as cur:
                    cur.execute("""
                        INSERT INTO evaluations (
                            plan_id, success, steps_planned, agents_used,
                            tools_used, insights
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        evaluation_data.get("plan_id", ""),
                        evaluation_data.get("success", False),
                        evaluation_data.get("steps_planned", 0),
                        evaluation_data.get("agents_used", []),
                        evaluation_data.get("tools_used", []),
                        evaluation_data.get("insights", [])
                    ))
                    self._task_board_store.commit()
            except Exception as e:
                self.logger.error(f"Evaluation storage failed: {str(e)}")
    
    async def _store_in_vector_memory(self, content_type: str, data: Dict[str, Any]):
        """Store data in long-term vector memory for semantic retrieval"""
        
        # Create searchable text representation
        if content_type == "interaction":
            content_text = f"User: {data.get('user_request', '')} Intent: {data.get('intent', '')} Success: {data.get('success', False)}"
        else:
            content_text = json.dumps(data, default=str)
        
        content_hash = hashlib.sha256(content_text.encode()).hexdigest()
        
        if self._vector_store == "pgvector":
            try:
                # Generate embedding (you'd use your actual embedding model here)
                embedding = await self._generate_embedding(content_text)
                
                with self._task_board_store.cursor() as cur:
                    cur.execute("""
                        INSERT INTO memory_embeddings (
                            content_hash, content_type, content_text, 
                            embedding, metadata, user_id
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (content_hash) DO NOTHING
                    """, (
                        content_hash,
                        content_type,
                        content_text,
                        embedding,
                        json.dumps(data),
                        data.get("user_id", "default")
                    ))
                    self._task_board_store.commit()
            except Exception as e:
                self.logger.error(f"pgvector storage failed: {str(e)}")
        
        elif self._vector_store == "chroma":
            try:
                if content_type == "interaction":
                    collection = self._interactions_collection
                else:
                    collection = self._preferences_collection
                
                collection.add(
                    documents=[content_text],
                    metadatas=[data],
                    ids=[content_hash]
                )
            except Exception as e:
                self.logger.error(f"Chroma storage failed: {str(e)}")
    
    async def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search long-term memory for relevant context"""
        
        results = []
        
        if self._vector_store == "pgvector":
            try:
                query_embedding = await self._generate_embedding(query)
                
                with self._task_board_store.cursor() as cur:
                    cur.execute("""
                        SELECT content_text, metadata, 
                               1 - (embedding <=> %s) as similarity
                        FROM memory_embeddings
                        WHERE 1 - (embedding <=> %s) > 0.7
                        ORDER BY embedding <=> %s
                        LIMIT %s
                    """, (query_embedding, query_embedding, query_embedding, limit))
                    
                    for row in cur.fetchall():
                        results.append({
                            "content": row[0],
                            "metadata": row[1],
                            "similarity": row[2]
                        })
            except Exception as e:
                self.logger.error(f"pgvector search failed: {str(e)}")
        
        elif self._vector_store == "chroma":
            try:
                search_results = self._interactions_collection.query(
                    query_texts=[query],
                    n_results=limit
                )
                
                if search_results['documents'] and search_results['documents'][0]:
                    for i, doc in enumerate(search_results['documents'][0]):
                        metadata = search_results['metadatas'][0][i] if search_results['metadatas'] else {}
                        distance = search_results['distances'][0][i] if search_results['distances'] else 1.0
                        
                        results.append({
                            "content": doc,
                            "metadata": metadata,
                            "similarity": 1.0 - distance  # Convert distance to similarity
                        })
            except Exception as e:
                self.logger.error(f"Chroma search failed: {str(e)}")
        
        return results
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (placeholder - use your actual embedding model)"""
        
        # This is where you'd integrate with your actual embedding model
        # For now, return a dummy embedding
        import random
        return [random.random() for _ in range(1536)]  # OpenAI embedding size
    
    def _generate_task_id(self, task_data: Dict[str, Any]) -> str:
        """Generate unique task ID"""
        content = f"{task_data.get('title', '')}{task_data.get('description', '')}{datetime.utcnow().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_interaction_id(self, interaction_data: Dict[str, Any]) -> str:
        """Generate unique interaction ID"""
        content = f"{interaction_data.get('user_request', '')}{datetime.utcnow().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def cleanup_expired_data(self):
        """Clean up expired data from memory tiers"""
        
        # Clean scratchpad (handled by Redis TTL or in-memory expiry)
        if not self._scratchpad_store:
            current_time = datetime.utcnow()
            expired_keys = [
                key for key, value in self._scratchpad_memory.items()
                if current_time >= value["expires_at"]
            ]
            for key in expired_keys:
                del self._scratchpad_memory[key]
        
        # Clean old completed tasks (keep for 30 days)
        if self._task_board_store:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                with self._task_board_store.cursor() as cur:
                    cur.execute("""
                        DELETE FROM interactions 
                        WHERE timestamp < %s
                    """, (cutoff_date,))
                    self._task_board_store.commit()
            except Exception as e:
                self.logger.error(f"Cleanup failed: {str(e)}") 