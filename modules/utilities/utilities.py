from typing import List, Dict, Any, Optional, Union, Tuple, cast, TypeVar, Type
from pathlib import Path
import logging
from datetime import datetime
import json
import hashlib
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.pydantic_v1 import BaseModel, Field
from chromadb import Client, Collection
import chromadb.utils.embedding_functions as embedding_functions
import streamlit as st

# Constants
DEFAULT_CHROMA_PERSIST_DIR = Path("chroma_db")
DEFAULT_CHROMA_COLLECTION_NAME = "document_collection"
DEFAULT_CACHE_DIR = Path(".cache")
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_MAX_WORKERS = 4
DEFAULT_DOCUMENT_CHUNK_SIZE = 1000
DEFAULT_DOCUMENT_CHUNK_OVERLAP = 200

# Type aliases
T = TypeVar('T')
MessageLike = Union[BaseMessage, Dict[str, Any], str]
DocumentLike = Union[Document, Dict[str, Any], str]

class ChatMessageDict(TypedDict):
    """Typed dictionary for chat messages."""
    content: str
    sender: str
    timestamp: str
    message_id: str

class RetrievalConfig(BaseModel):
    """Configuration for document retrieval."""
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    score_threshold: float = Field(default=0.7, description="Minimum similarity score to consider")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")

class EmbeddingConfig(BaseModel):
    """Configuration for embeddings."""
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    dimensions: Optional[int] = Field(default=None, description="Embedding dimensions")

class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    persist_directory: Path = Field(default=DEFAULT_CHROMA_PERSIST_DIR, description="Directory to persist vector store")
    collection_name: str = Field(default=DEFAULT_CHROMA_COLLECTION_NAME, description="Collection name")
    embedding_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig, description="Embedding configuration")

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    log_level: int = Field(default=DEFAULT_LOG_LEVEL, description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Path to log file")

class CacheConfig(BaseModel):
    """Configuration for caching."""
    cache_dir: Path = Field(default=DEFAULT_CACHE_DIR, description="Directory for cache files")
    max_size_mb: int = Field(default=100, description="Maximum cache size in MB")

class AppConfig(BaseModel):
    """Main application configuration."""
    vector_store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    max_workers: int = Field(default=DEFAULT_MAX_WORKERS, description="Maximum workers for parallel processing")

def setup_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        config: Optional logging configuration. If None, uses default configuration.

    Returns:
        Configured logger instance.
    """
    if config is None:
        config = LoggingConfig()

    logger = logging.getLogger("chatbot")
    logger.setLevel(config.log_level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log file is specified
    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(config.log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_timestamp() -> str:
    """
    Get current timestamp as ISO format string.

    Returns:
        Current timestamp in ISO format.
    """
    return datetime.now().isoformat()

def generate_message_id(content: str, sender: str) -> str:
    """
    Generate a unique message ID based on content and sender.

    Args:
        content: Message content
        sender: Message sender

    Returns:
        Unique message ID
    """
    unique_str = f"{content}-{sender}-{get_timestamp()}"
    return hashlib.sha256(unique_str.encode()).hexdigest()

def convert_to_chat_message(message: MessageLike) -> ChatMessageDict:
    """
    Convert various message formats to standardized ChatMessageDict.

    Args:
        message: Message in various formats (BaseMessage, dict, or str)

    Returns:
        Standardized ChatMessageDict

    Raises:
        ValueError: If message format is invalid
    """
    if isinstance(message, BaseMessage):
        return {
            "content": message.content,
            "sender": message.type,
            "timestamp": get_timestamp(),
            "message_id": generate_message_id(message.content, message.type)
        }
    elif isinstance(message, dict):
        required_keys = {"content", "sender"}
        if not required_keys.issubset(message.keys()):
            raise ValueError(f"Message dict must contain keys: {required_keys}")
        return {
            "content": str(message["content"]),
            "sender": str(message["sender"]),
            "timestamp": message.get("timestamp", get_timestamp()),
            "message_id": message.get("message_id", generate_message_id(str(message["content"]), str(message["sender"])))
        }
    elif isinstance(message, str):
        return {
            "content": message,
            "sender": "user",
            "timestamp": get_timestamp(),
            "message_id": generate_message_id(message, "user")
        }
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")

def convert_chat_messages_to_list(messages: List[MessageLike]) -> List[ChatMessageDict]:
    """
    Convert a list of messages to standardized ChatMessageDict format.

    Args:
        messages: List of messages in various formats

    Returns:
        List of standardized ChatMessageDict
    """
    return [convert_to_chat_message(msg) for msg in messages]

def get_vector_store(config: Optional[VectorStoreConfig] = None) -> VectorStore:
    """
    Initialize and return a Chroma vector store.

    Args:
        config: Optional vector store configuration. If None, uses default configuration.

    Returns:
        Initialized vector store instance.
    """
    if config is None:
        config = VectorStoreConfig()

    # Create persist directory if it doesn't exist
    config.persist_directory.mkdir(parents=True, exist_ok=True)

    # Initialize Chroma client
    chroma_client = Client(
        path=str(config.persist_directory.absolute())
    )

    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

    # Create and return vector store
    return Chroma(
        collection_name=config.collection_name,
        persist_directory=str(config.persist_directory.absolute()),
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

def get_embedding_function(config: Optional[EmbeddingConfig] = None) -> Embeddings:
    """
    Get embedding function based on configuration.

    Args:
        config: Optional embedding configuration. If None, uses default configuration.

    Returns:
        Embedding function instance.
    """
    if config is None:
        config = EmbeddingConfig()

    return embedding_functions.DefaultEmbeddingFunction(
        model_name=config.model_name
    )

def get_retriever(
    vector_store: VectorStore,
    config: Optional[RetrievalConfig] = None
) -> BaseRetriever:
    """
    Create a retriever from a vector store with optional configuration.

    Args:
        vector_store: Vector store instance
        config: Optional retrieval configuration

    Returns:
        Configured retriever instance
    """
    if config is None:
        config = RetrievalConfig()

    return vector_store.as_retriever(
        search_kwargs={
            "k": config.top_k,
            "score_threshold": config.score_threshold,
            **({"filter": config.filter} if config.filter else {})
        }
    )

def format_docs(docs: List[Document]) -> str:
    """
    Format a list of documents into a single string.

    Args:
        docs: List of Document objects

    Returns:
        Formatted string containing all document contents
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_prompt_template(template: str, input_variables: List[str]) -> PromptTemplate:
    """
    Create a prompt template from a string template.

    Args:
        template: String template with placeholders
        input_variables: List of variable names used in the template

    Returns:
        Configured PromptTemplate instance
    """
    return PromptTemplate(
        template=template,
        input_variables=input_variables
    )

def get_model_response(
    model: BaseLanguageModel,
    prompt: str,
    config: Optional[RunnableConfig] = None
) -> str:
    """
    Get response from a language model.

    Args:
        model: Language model instance
        prompt: Input prompt
        config: Optional runnable configuration

    Returns:
        Model response as string
    """
    if config is None:
        config = RunnableConfig()

    response = model.invoke(prompt, config=config)
    return response.content if hasattr(response, 'content') else str(response)

def save_to_cache(data: Any, key: str, config: Optional[CacheConfig] = None) -> bool:
    """
    Save data to cache with a given key.

    Args:
        data: Data to cache
        key: Cache key
        config: Optional cache configuration

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = CacheConfig()

    try:
        config.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = config.cache_dir / f"{key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        st.error(f"Failed to save to cache: {e}")
        return False

def load_from_cache(key: str, config: Optional[CacheConfig] = None) -> Optional[Any]:
    """
    Load data from cache with a given key.

    Args:
        key: Cache key
        config: Optional cache configuration

    Returns:
        Cached data if found, None otherwise
    """
    if config is None:
        config = CacheConfig()

    cache_file = config.cache_dir / f"{key}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load from cache: {e}")
        return None

def clear_cache(config: Optional[CacheConfig] = None) -> bool:
    """
    Clear the entire cache directory.

    Args:
        config: Optional cache configuration

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = CacheConfig()

    try:
        if config.cache_dir.exists():
            for item in config.cache_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    import shutil
                    shutil.rmtree(item)

        return True
    except Exception as e:
        st.error(f"Failed to clear cache: {e}")
        return False

def validate_document(document: DocumentLike) -> Document:
    """
    Validate and convert document to Document format.

    Args:
        document: Document in various formats

    Returns:
        Validated Document instance

    Raises:
        ValueError: If document format is invalid
    """
    if isinstance(document, Document):
        return document
    elif isinstance(document, dict):
        return Document(
            page_content=document.get("page_content", ""),
            metadata=document.get("metadata", {})
        )
    elif isinstance(document, str):
        return Document(
            page_content=document,
            metadata={}
        )
    else:
        raise ValueError(f"Unsupported document type: {type(document)}")

def batch_process_documents(
    documents: List[DocumentLike],
    process_func: callable,
    max_workers: int = DEFAULT_MAX_WORKERS
) -> List[Any]:
    """
    Process documents in parallel batches.

    Args:
        documents: List of documents to process
        process_func: Function to process each document
        max_workers: Maximum number of workers for parallel processing

    Returns:
        List of processed results
    """
    validated_docs = [validate_document(doc) for doc in documents]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, doc) for doc in validated_docs]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                st.error(f"Error processing document: {e}")
                results.append(None)

    return results

def get_chat_history(messages: List[ChatMessageDict]) -> List[BaseMessage]:
    """
    Convert chat messages to LangChain BaseMessage format.

    Args:
        messages: List of chat messages

    Returns:
        List of BaseMessage instances
    """
    history = []
    for msg in messages:
        if msg["sender"] == "human":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["sender"] == "ai":
            history.append(AIMessage(content=msg["content"]))
        elif msg["sender"] == "system":
            history.append(SystemMessage(content=msg["content"]))
    return history

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    import re
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Limit length
    return sanitized[:255]

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.

    Args:
        filename: Input filename

    Returns:
        File extension (including dot)
    """
    return Path(filename).suffix.lower()

def is_supported_file(filename: str) -> bool:
    """
    Check if a file is supported by the application.

    Args:
        filename: Input filename

    Returns:
        True if file is supported, False otherwise
    """
    supported_extensions = {
        '.txt', '.pdf', '.docx', '.md', '.csv', '.json', '.html', '.htm'
    }
    return get_file_extension(filename) in supported_extensions

class Chroma(VectorStore):
    """
    Custom Chroma vector store implementation with additional functionality.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME,
        persist_directory: Optional[Union[str, Path]] = None,
        embedding_function: Optional[Embeddings] = None
    ):
        """
        Initialize Chroma vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            embedding_function: Embedding function to use
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory) if persist_directory else DEFAULT_CHROMA_PERSIST_DIR
        self.embedding_function = embedding_function or embedding_functions.DefaultEmbeddingFunction()

        # Initialize Chroma client
        self.client = Client(path=str(self.persist_directory.absolute()))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
            kwargs: Additional arguments

        Returns:
            List of document IDs
        """
        # Convert documents to format expected by Chroma
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(i) for i in range(len(documents))]

        # Add to collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            kwargs: Additional arguments

        Returns:
            List of similar documents
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter
        )

        documents = []
        for i, (text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            documents.append(Document(
                page_content=text,
                metadata=metadata or {}
            ))

        return documents

    def persist(self) -> None:
        """
        Persist the vector store to disk.
        """
        # Chroma persists automatically, but we can force it
        self.client.persist()

    def delete_collection(self) -> None:
        """
        Delete the collection from the vector store.
        """
        self.client.delete_collection(self.collection_name)
        self.collection = None

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """
        Get the embedding function.

        Returns:
            Embedding function or None
        """
        return self.embedding_function

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """
        Create a retriever from the vector store.

        Args:
            kwargs: Additional arguments for retriever

        Returns:
            BaseRetriever instance
        """
        from langchain_community.retrievers import ChromaRetriever

        return ChromaRetriever(
            collection=self.collection,
            **kwargs
        )