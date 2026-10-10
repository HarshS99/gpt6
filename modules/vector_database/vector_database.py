from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
class VectorStoreError(Exception): pass
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    A wrapper class for ChromaDB vector database operations.
    Handles document ingestion, embedding, storage, and retrieval.
    """

    def __init__(
        self,
        persist_directory: Union[str, Path] = "chroma_db",
        collection_name: str = "document_collection",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Initialize the VectorDatabase with configuration.

        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection to store documents
            embedding_model_name: Name of the HuggingFace embedding model
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between text chunks
            **kwargs: Additional keyword arguments for ChromaDB settings
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs

        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Initialize persistent client
        self.client = PersistentClient(path=str(self.persist_directory))

        # Initialize vector store
        self.vector_store: Optional[VectorStore] = None

    def initialize_collection(self) -> None:
        """Initialize or load the ChromaDB collection."""
        try:
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Initialized collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise VectorStoreError(f"Failed to initialize collection: {e}")

    def add_documents(
        self,
        documents: List[Document],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Add documents to the vector database.

        Args:
            documents: List of LangChain Document objects
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
            **kwargs: Additional arguments for the vector store

        Returns:
            List of document IDs that were added
        """
        if not self.vector_store:
            self.initialize_collection()

        try:
            document_ids = self.vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                **kwargs
            )
            logger.info(f"Added {len(document_ids)} documents to collection")
            return document_ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}")

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Add raw texts to the vector database.

        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
            **kwargs: Additional arguments for the vector store

        Returns:
            List of document IDs that were added
        """
        if not self.vector_store:
            self.initialize_collection()

        try:
            document_ids = self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                **kwargs
            )
            logger.info(f"Added {len(document_ids)} texts to collection")
            return document_ids
        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            raise VectorStoreError(f"Failed to add texts: {e}")

    def add_documents_from_path(
        self,
        file_path: Union[str, Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Add documents from a file path, automatically splitting and embedding.

        Args:
            file_path: Path to the file to load
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            **kwargs: Additional arguments for document loading

        Returns:
            List of document IDs that were added
        """
        if not self.vector_store:
            self.initialize_collection()

        try:
            # Use provided chunk size or fall back to instance value
            split_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
            split_chunk_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

            # Load and split documents
            from langchain_community.document_loaders import (
                TextLoader,
                PyPDFLoader,
                Docx2txtLoader,
                UnstructuredPowerPointLoader,
                CSVLoader
            )

            path = Path(file_path)
            loader = None

            # Select appropriate loader based on file extension
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path), **kwargs)
            elif path.suffix.lower() == ".txt":
                loader = TextLoader(str(path), **kwargs)
            elif path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(path), **kwargs)
            elif path.suffix.lower() in (".ppt", ".pptx"):
                loader = UnstructuredPowerPointLoader(str(path), **kwargs)
            elif path.suffix.lower() == ".csv":
                loader = CSVLoader(str(path), **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

            # Load documents
            raw_documents = loader.load()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=split_chunk_size,
                chunk_overlap=split_chunk_overlap,
                length_function=len,
            )
            documents = text_splitter.split_documents(raw_documents)

            # Add to vector store
            document_ids = self.add_documents(documents)
            logger.info(f"Added {len(document_ids)} documents from {path}")
            return document_ids

        except Exception as e:
            logger.error(f"Failed to add documents from path {file_path}: {e}")
            raise VectorStoreError(f"Failed to add documents from path: {e}")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any]
    ) -> List[Document]:
        """
        Perform similarity search in the vector database.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional arguments for similarity search

        Returns:
            List of similar documents
        """
        if not self.vector_store:
            self.initialize_collection()

        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
            return results
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise VectorStoreError(f"Failed to perform similarity search: {e}")

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional arguments for similarity search

        Returns:
            List of tuples containing (document, score)
        """
        if not self.vector_store:
            self.initialize_collection()

        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
            return results
        except Exception as e:
            logger.error(f"Failed to perform similarity search with score: {e}")
            raise VectorStoreError(f"Failed to perform similarity search with score: {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        Returns:
            Dictionary containing collection information
        """
        if not self.vector_store:
            self.initialize_collection()

        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": collection.name,
                "metadata": collection.metadata,
                "count": collection.count(),
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise VectorStoreError(f"Failed to get collection info: {e}")

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        if not self.vector_store:
            self.initialize_collection()

        try:
            collection = self.client.get_collection(self.collection_name)
            collection.delete(where={})  # Empty filter deletes all documents
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise VectorStoreError(f"Failed to clear collection: {e}")

    def persist(self) -> None:
        """Persist the vector database to disk."""
        if not self.vector_store:
            self.initialize_collection()

        try:
            self.vector_store.persist()
            logger.info(f"Persisted vector database to {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to persist vector database: {e}")
            raise VectorStoreError(f"Failed to persist vector database: {e}")

    def delete_collection(self) -> None:
        """Delete the entire collection from disk."""
        try:
            # Close the persistent client
            self.client.clear_system_cache()
            self.client.reset()

            # Remove the collection directory
            import shutil
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)

            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.initialize_collection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.persist()
        self.client.clear_system_cache()
        self.client.reset()