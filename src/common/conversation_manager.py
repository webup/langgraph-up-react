"""Conversation manager that integrates with LangGraph ReAct agent."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage, HumanMessage

from react_agent.state import State

from .context import Context
from .conversation import (
    ConversationSession,
    ConversationStorage,
    FileStorage,
    HistoryManager,
    MemoryStorage,
    generate_session_id,
)


class ConversationManager:
    """Manages multi-turn conversations with the ReAct agent."""
    
    def __init__(
        self,
        storage: Optional[ConversationStorage] = None,
        history_manager: Optional[HistoryManager] = None,
        auto_save: bool = True,
    ):
        self.storage = storage or MemoryStorage()
        self.history_manager = history_manager or HistoryManager()
        self.auto_save = auto_save
        self._active_sessions: Dict[str, ConversationSession] = {}
    
    async def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session."""
        if session_id is None:
            session_id = generate_session_id()
        
        session = ConversationSession(id=session_id)
        self._active_sessions[session_id] = session
        
        if self.auto_save:
            await self.storage.save_session(session)
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session."""
        # Check active sessions first
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]
        
        # Load from storage
        session = await self.storage.load_session(session_id)
        if session:
            self._active_sessions[session_id] = session
        
        return session
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        # Remove from active sessions
        self._active_sessions.pop(session_id, None)
        
        # Delete from storage
        return await self.storage.delete_session(session_id)
    
    async def list_sessions(self) -> List[str]:
        """List all conversation sessions."""
        return await self.storage.list_sessions()
    
    async def add_message(self, session_id: str, message: BaseMessage) -> None:
        """Add a message to a session."""
        session = await self.get_session(session_id)
        if session:
            session.add_message(message)
            
            if self.auto_save:
                await self.storage.save_session(session)
    
    async def get_messages(self, session_id: str) -> List[BaseMessage]:
        """Get all messages from a session."""
        session = await self.get_session(session_id)
        if session:
            return session.messages
        return []
    
    async def prepare_state_for_graph(self, session_id: str) -> Dict[str, Any]:
        """Prepare state for LangGraph execution."""
        messages = await self.get_messages(session_id)
        
        # Apply history management
        if self.history_manager.should_compress_history(messages):
            messages = self.history_manager.compress_history(messages)
        
        return {"messages": messages}
    
    async def update_session_from_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """Update session with new state from LangGraph execution."""
        session = await self.get_session(session_id)
        if session and "messages" in state:
            # Get new messages (those not already in the session)
            new_messages = state["messages"][len(session.messages):]
            
            for message in new_messages:
                session.add_message(message)
            
            if self.auto_save:
                await self.storage.save_session(session)


class ChatInterface:
    """User-friendly interface for multi-turn conversations."""
    
    def __init__(
        self,
        conversation_manager: Optional[ConversationManager] = None,
        default_context: Optional[Context] = None,
    ):
        self.conversation_manager = conversation_manager or ConversationManager()
        self.default_context = default_context or Context()
    
    async def start_conversation(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session."""
        return await self.conversation_manager.create_session(session_id)
    
    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[Context] = None,
    ) -> str:
        """Send a message and get a response."""
        # Import here to avoid circular imports
        from react_agent import graph
        
        # Create session if not provided
        if session_id is None:
            session_id = await self.start_conversation()
        
        # Ensure session exists
        session = await self.conversation_manager.get_session(session_id)
        if session is None:
            session_id = await self.start_conversation()
        
        # Add user message to session
        user_message = HumanMessage(content=message)
        await self.conversation_manager.add_message(session_id, user_message)
        
        # Prepare state for graph
        state = await self.conversation_manager.prepare_state_for_graph(session_id)
        
        # Execute graph
        context = context or self.default_context
        result = await graph.ainvoke(state, context=context)
        
        # Update session with results
        await self.conversation_manager.update_session_from_state(session_id, result)
        
        # Return the last AI message
        if result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                return str(last_message.content)
        
        return "I apologize, but I couldn't generate a response."
    
    async def stream_chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[Context] = None,
    ) -> AsyncGenerator[str, None]:
        """Send a message and stream the response."""
        # Import here to avoid circular imports
        from react_agent import graph
        
        # Create session if not provided
        if session_id is None:
            session_id = await self.start_conversation()
        
        # Ensure session exists
        session = await self.conversation_manager.get_session(session_id)
        if session is None:
            session_id = await self.start_conversation()
        
        # Add user message to session
        user_message = HumanMessage(content=message)
        await self.conversation_manager.add_message(session_id, user_message)
        
        # Prepare state for graph
        state = await self.conversation_manager.prepare_state_for_graph(session_id)
        
        # Stream graph execution
        context = context or self.default_context
        final_state = None
        
        async for chunk in graph.astream(state, context=context):
            for node_name, node_output in chunk.items():
                if node_name == "call_model" and "messages" in node_output:
                    message = node_output["messages"][-1]
                    if hasattr(message, 'content') and message.content:
                        yield str(message.content)
                        final_state = chunk
        
        # Update session with final results if available
        if final_state:
            # Reconstruct the full state from the final chunk
            all_messages = await self.conversation_manager.get_messages(session_id)
            for node_output in final_state.values():
                if "messages" in node_output:
                    # Add new messages that aren't already in the session
                    for msg in node_output["messages"]:
                        if msg not in all_messages:
                            await self.conversation_manager.add_message(session_id, msg)
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history in a readable format."""
        messages = await self.conversation_manager.get_messages(session_id)
        
        history = []
        for msg in messages:
            history.append({
                "role": msg.type,
                "content": str(msg.content),
                "timestamp": getattr(msg, 'timestamp', None),
            })
        
        return history
    
    async def clear_conversation(self, session_id: str) -> bool:
        """Clear a conversation session."""
        return await self.conversation_manager.delete_session(session_id)
    
    async def list_conversations(self) -> List[str]:
        """List all conversation sessions."""
        return await self.conversation_manager.list_sessions()


# Convenience functions for easy usage
_default_chat_interface: Optional[ChatInterface] = None


def get_default_chat_interface() -> ChatInterface:
    """Get the default chat interface (singleton pattern)."""
    global _default_chat_interface
    if _default_chat_interface is None:
        # Use file storage for persistence across sessions
        storage = FileStorage()
        conversation_manager = ConversationManager(storage=storage)
        _default_chat_interface = ChatInterface(conversation_manager=conversation_manager)
    return _default_chat_interface


async def quick_chat(message: str, session_id: Optional[str] = None) -> tuple[str, str]:
    """Quick chat function that returns (response, session_id)."""
    interface = get_default_chat_interface()
    
    if session_id is None:
        session_id = await interface.start_conversation()
    
    response = await interface.chat(message, session_id)
    return response, session_id


async def quick_stream_chat(
    message: str, session_id: Optional[str] = None
) -> tuple[AsyncGenerator[str, None], str]:
    """Quick streaming chat function that returns (stream, session_id)."""
    interface = get_default_chat_interface()
    
    if session_id is None:
        session_id = await interface.start_conversation()
    
    stream = interface.stream_chat(message, session_id)
    return stream, session_id