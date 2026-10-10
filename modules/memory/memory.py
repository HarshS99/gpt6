from typing import List, Dict, Any, Optional, Union, Tuple, cast
from pathlib import Path
import logging
from datetime import datetime
try:
    from langchain_core.memory import BaseMemory
except Exception:
    class BaseMemory:
        """Lightweight fallback for langchain_core.memory.BaseMemory.

        This provides the minimal interface used by ConversationMemory in this
        project so the app can run when the installed langchain_core package
        lacks the `memory` module.
        """
        def __init__(self):
            pass

        def __getattr__(self, name):
            raise AttributeError(f"BaseMemory fallback has no attribute '{name}'")

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ConversationMemoryConfig(BaseModel):
    """Configuration for conversation memory management."""
    max_history: int = Field(default=10, description="Maximum number of messages to keep in history")
    memory_key: str = Field(default="chat_history", description="Key to use for memory in prompt")
    return_messages: bool = Field(default=True, description="Whether to return messages or content strings")

class ConversationMemory(BaseMemory):
    """
    Memory implementation for chatbot conversations using LangChain's BaseMemory.
    Handles storing, retrieving, and managing conversation history.
    """

    def __init__(
        self,
        session_id: str,
        config: Optional[ConversationMemoryConfig] = None,
        chat_history: Optional[BaseChatMessageHistory] = None
    ):
        """
        Initialize conversation memory.

        Args:
            session_id: Unique identifier for the conversation session
            config: Configuration for memory management
            chat_history: Optional existing chat history to use
        """
        super().__init__()
        self.session_id = session_id
        self.config = config if config else ConversationMemoryConfig()
        self.chat_history = chat_history if chat_history else ChatMessageHistory()
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize the memory system."""
        self._is_initialized = True
        logger.info(f"Initialized conversation memory for session: {self.session_id}")

    def clear(self) -> None:
        """Clear the conversation history."""
        self.chat_history.clear()
        logger.info(f"Cleared conversation history for session: {self.session_id}")

    @property
    def memory_variables(self) -> List[str]:
        """Return list of memory variables."""
        return [self.config.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables from chat history.

        Args:
            inputs: Dictionary of input variables

        Returns:
            Dictionary containing memory variables
        """
        if not self._is_initialized:
            self.initialize()

        messages = self.chat_history.messages
        if not messages:
            return {self.config.memory_key: []}

        # Apply history limit
        history = messages[-self.config.max_history:] if self.config.max_history > 0 else messages

        if self.config.return_messages:
            return {self.config.memory_key: history}
        else:
            return {
                self.config.memory_key: [
                    {"role": msg.type, "content": msg.content}
                    for msg in history
                ]
            }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Save conversation context from inputs and outputs.

        Args:
            inputs: Dictionary of input messages
            outputs: Dictionary of output messages
        """
        if not self._is_initialized:
            self.initialize()

        # Extract human message from inputs
        human_message = None
        for key, value in inputs.items():
            if isinstance(value, (str, BaseMessage)):
                if isinstance(value, str):
                    human_message = HumanMessage(content=value)
                elif isinstance(value, HumanMessage):
                    human_message = value
                elif isinstance(value, BaseMessage):
                    human_message = HumanMessage(content=value.content)
                break

        # Extract AI message from outputs
        ai_message = None
        for key, value in outputs.items():
            if isinstance(value, (str, BaseMessage)):
                if isinstance(value, str):
                    ai_message = AIMessage(content=value)
                elif isinstance(value, AIMessage):
                    ai_message = value
                elif isinstance(value, BaseMessage):
                    ai_message = AIMessage(content=value.content)
                break

        # Add messages to history if they exist
        if human_message:
            self.chat_history.add_message(human_message)
        if ai_message:
            self.chat_history.add_message(ai_message)

        logger.debug(f"Saved conversation context for session: {self.session_id}")

    def get_chat_history(self) -> List[BaseMessage]:
        """
        Get the complete chat history.

        Returns:
            List of BaseMessage objects
        """
        return self.chat_history.messages

    def get_session_id(self) -> str:
        """
        Get the session ID.

        Returns:
            Session ID string
        """
        return self.session_id

    def get_memory_config(self) -> ConversationMemoryConfig:
        """
        Get the memory configuration.

        Returns:
            ConversationMemoryConfig object
        """
        return self.config

class SessionManager:
    """
    Manages multiple conversation sessions and their associated memories.
    """

    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, ConversationMemory] = {}
        self._default_session_id = "default_session"

    def create_session(self, session_id: Optional[str] = None) -> ConversationMemory:
        """
        Create a new conversation session.

        Args:
            session_id: Optional session ID. If None, uses default.

        Returns:
            ConversationMemory instance
        """
        session_id = session_id or self._default_session_id
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists. Returning existing session.")
            return self.sessions[session_id]

        memory = ConversationMemory(session_id=session_id)
        self.sessions[session_id] = memory
        logger.info(f"Created new session: {session_id}")
        return memory

    def get_session(self, session_id: Optional[str] = None) -> ConversationMemory:
        """
        Get an existing conversation session.

        Args:
            session_id: Session ID to retrieve

        Returns:
            ConversationMemory instance

        Raises:
            ValueError: If session does not exist
        """
        session_id = session_id or self._default_session_id
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} does not exist")
        return self.sessions[session_id]

    def delete_session(self, session_id: Optional[str] = None) -> bool:
        """
        Delete a conversation session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if it didn't exist
        """
        session_id = session_id or self._default_session_id
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def list_sessions(self) -> List[str]:
        """
        List all active session IDs.

        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())

    def clear_all_sessions(self) -> None:
        """Clear all conversation sessions."""
        self.sessions.clear()
        logger.info("Cleared all conversation sessions")

def create_memory_chain(
    llm: Any,
    memory: ConversationMemory,
    prompt: Optional[ChatPromptTemplate] = None
) -> RunnableWithMessageHistory:
    """
    Create a memory-aware chain with message history.

    Args:
        llm: Language model to use
        memory: ConversationMemory instance
        prompt: Optional custom prompt template

    Returns:
        RunnableWithMessageHistory instance
    """
    # Use default prompt if none provided
    if prompt is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer the user's questions based on the conversation history."),
            MessagesPlaceholder(variable_name=memory.config.memory_key),
            ("human", "{input}"),
        ])

    # Create the chain
    chain = prompt | llm

    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key=memory.config.memory_key,
    )

    return chain_with_history

def get_message_content(message: Union[str, BaseMessage]) -> str:
    """
    Extract content from a message (string or BaseMessage).

    Args:
        message: Message to extract content from

    Returns:
        String content
    """
    if isinstance(message, str):
        return message
    elif isinstance(message, BaseMessage):
        return message.content
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")


class MemoryManager:
    """Compatibility wrapper providing a simple memory manager API expected by the UI.

    This wraps `SessionManager` / `ConversationMemory` to offer:
      - `initialize_memory(session_id)`
      - `update_memory(session_id, user_message, ai_response)`
      - `get_memory(session_id)` -> ConversationMemory
      - `clear_memory(session_id)`
    """

    def __init__(self):
        self._manager = SessionManager()

    def initialize_memory(self, session_id: Optional[str]) -> None:
        sid = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._manager.create_session(sid)

    def update_memory(self, session_id: Optional[str], user_message: str, ai_response: str) -> None:
        sid = session_id or self._manager._default_session_id
        if sid not in self._manager.sessions:
            self._manager.create_session(sid)
        mem = self._manager.get_session(sid)
        mem.save_context({'input': user_message}, {'output': ai_response})

    def get_memory(self, session_id: Optional[str]) -> ConversationMemory:
        sid = session_id or self._manager._default_session_id
        if sid not in self._manager.sessions:
            return self._manager.create_session(sid)
        return self._manager.get_session(sid)

    def clear_memory(self, session_id: Optional[str]) -> None:
        sid = session_id or self._manager._default_session_id
        if sid in self._manager.sessions:
            self._manager.delete_session(sid)