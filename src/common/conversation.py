"""Multi-turn conversation management components."""

from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AnyMessage, BaseMessage


@dataclass
class ConversationSession:
    """Represents a conversation session with metadata."""
    
    id: str
    messages: List[BaseMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self.messages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "id": self.id,
            "messages": [msg.dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationSession:
        """Create session from dictionary."""
        from langchain_core.messages import messages_from_dict
        
        session = cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )
        
        # Reconstruct messages
        if "messages" in data and data["messages"]:
            session.messages = messages_from_dict(data["messages"])
        
        return session


class ConversationStorage(ABC):
    """Abstract base class for conversation storage backends."""
    
    @abstractmethod
    async def save_session(self, session: ConversationSession) -> None:
        """Save a conversation session."""
        pass
    
    @abstractmethod
    async def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load a conversation session by ID."""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        pass
    
    @abstractmethod
    async def list_sessions(self) -> List[str]:
        """List all session IDs."""
        pass


class MemoryStorage(ConversationStorage):
    """In-memory storage for conversations (non-persistent)."""
    
    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}
    
    async def save_session(self, session: ConversationSession) -> None:
        """Save session to memory."""
        self._sessions[session.id] = session
    
    async def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from memory."""
        return self._sessions.get(session_id)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    async def list_sessions(self) -> List[str]:
        """List all session IDs in memory."""
        return list(self._sessions.keys())


class FileStorage(ConversationStorage):
    """File-based storage for conversations."""
    
    def __init__(self, storage_dir: str = "./conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get file path for a session."""
        return self.storage_dir / f"{session_id}.json"
    
    async def save_session(self, session: ConversationSession) -> None:
        """Save session to file."""
        file_path = self._get_session_file(session.id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
    
    async def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from file."""
        file_path = self._get_session_file(session_id)
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ConversationSession.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session file."""
        file_path = self._get_session_file(session_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    async def list_sessions(self) -> List[str]:
        """List all session IDs from files."""
        return [
            f.stem for f in self.storage_dir.glob("*.json")
            if f.is_file()
        ]


class HistoryManager:
    """Manages conversation history with token limits and summarization."""
    
    def __init__(self, max_messages: int = 50, max_tokens: int = 4000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
    
    def should_compress_history(self, messages: Sequence[BaseMessage]) -> bool:
        """Check if history should be compressed."""
        if len(messages) > self.max_messages:
            return True
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        total_chars = sum(len(str(msg.content)) for msg in messages)
        estimated_tokens = total_chars // 4
        
        return estimated_tokens > self.max_tokens
    
    def compress_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Compress history by keeping recent messages and important context."""
        if not self.should_compress_history(messages):
            return messages
        
        # Keep first message (usually system prompt or important context)
        # Keep last N messages for recent context
        keep_recent = min(20, self.max_messages // 2)
        
        if len(messages) <= keep_recent + 1:
            return messages
        
        compressed = []
        
        # Keep first message if it exists
        if messages:
            compressed.append(messages[0])
        
        # Add summary indicator
        from langchain_core.messages import SystemMessage
        summary_msg = SystemMessage(
            content=f"[Previous conversation compressed - {len(messages) - keep_recent - 1} messages summarized]"
        )
        compressed.append(summary_msg)
        
        # Keep recent messages
        compressed.extend(messages[-keep_recent:])
        
        return compressed
    
    async def summarize_history(self, messages: List[BaseMessage]) -> str:
        """Generate a summary of conversation history (placeholder for LLM-based summarization)."""
        # This is a simple implementation
        # In production, you'd use an LLM to generate meaningful summaries
        
        user_messages = [msg for msg in messages if msg.type == "human"]
        ai_messages = [msg for msg in messages if msg.type == "ai"]
        
        summary = f"Conversation summary: {len(user_messages)} user messages, {len(ai_messages)} AI responses"
        
        if user_messages:
            first_user_msg = str(user_messages[0].content)[:100]
            summary += f". Started with: {first_user_msg}..."
        
        return summary


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())