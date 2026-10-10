from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    A modular RAG (Retrieval-Augmented Generation) pipeline that handles document ingestion,
    chunking, embedding, storage, and retrieval for question answering systems.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        persist_directory: Union[str, Path] = "chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedding_model: LangChain embedding model to use
            persist_directory: Directory to persist ChromaDB vectors
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve
        """
        self.embedding_model = embedding_model
        self.persist_directory = Path(persist_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.vectorstore: Optional[VectorStore] = None
        self.retriever: Optional[BaseRetriever] = None
        self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> None:
        """Initialize or load the Chroma vector store."""
        try:
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embedding_model,
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def load_documents(
        self,
        file_paths: List[Union[str, Path]],
        urls: Optional[List[str]] = None,
        file_type: str = "auto",
    ) -> List[Document]:
        """
        Load documents from various sources (files, URLs).

        Args:
            file_paths: List of file paths to load
            urls: List of URLs to load
            file_type: Type of files ('pdf', 'docx', 'txt', 'web', or 'auto')

        Returns:
            List of loaded documents
        """
        documents: List[Document] = []

        # Load files
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue

            try:
                if file_type == "pdf" or (file_type == "auto" and path.suffix.lower() == ".pdf"):
                    loader = PyPDFLoader(str(path))
                elif file_type == "docx" or (file_type == "auto" and path.suffix.lower() == ".docx"):
                    loader = Docx2txtLoader(str(path))
                elif file_type == "txt" or (file_type == "auto" and path.suffix.lower() == ".txt"):
                    loader = TextLoader(str(path))
                else:
                    logger.warning(f"Unsupported file type: {path}")
                    continue

                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {path}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                continue

        # Load URLs
        if urls:
            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from URL: {url}")
                except Exception as e:
                    logger.error(f"Failed to load URL {url}: {e}")
                    continue

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for better retrieval.

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        if not documents:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs

    def ingest_documents(
        self,
        file_paths: Optional[List[Union[str, Path]]] = None,
        urls: Optional[List[str]] = None,
        file_type: str = "auto",
    ) -> None:
        """
        Ingest documents into the vector store.

        Args:
            file_paths: List of file paths to ingest
            urls: List of URLs to ingest
            file_type: Type of files to ingest
        """
        documents = []

        # Load documents
        if file_paths:
            file_docs = self.load_documents(file_paths, file_type=file_type)
            documents.extend(file_docs)

        if urls:
            url_docs = self.load_documents([], urls=urls, file_type="web")
            documents.extend(url_docs)

        if not documents:
            logger.warning("No documents to ingest")
            return

        # Split documents
        split_docs = self.split_documents(documents)

        # Add to vector store
        try:
            self.vectorstore.add_documents(split_docs)
            self.vectorstore.persist()
            logger.info(f"Ingested {len(split_docs)} document chunks into vector store")
        except Exception as e:
            logger.error(f"Failed to ingest documents: {e}")
            raise

    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The query to search for

        Returns:
            List of relevant documents
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call ingest_documents first.")

        try:
            docs = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(docs)} documents for query: {query}")
            return docs
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise

    def create_retrieval_chain(
        self,
        prompt_template: str,
        output_parser: Optional[StrOutputParser] = None,
    ) -> RunnablePassthrough:
        """
        Create a retrieval chain for question answering.

        Args:
            prompt_template: The prompt template to use
            output_parser: Optional output parser

        Returns:
            A RunnablePassthrough chain combining retrieval and prompt
        """
        prompt = PromptTemplate.from_template(prompt_template)

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context": lambda x: format_docs(self.retrieve_documents(x["question"])),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.embedding_model
            | (output_parser if output_parser else StrOutputParser())
        )

        return chain

    def clear_vectorstore(self) -> None:
        """Clear all documents from the vector store."""
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = None
            self.retriever = None
            logger.info("Vector store cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            raise

    def get_vectorstore_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        if not self.vectorstore:
            return {"status": "not initialized"}

        try:
            collection = self.vectorstore._collection
            return {
                "status": "initialized",
                "persist_directory": str(self.persist_directory),
                "num_documents": collection.count(),
                "embedding_model": self.embedding_model.__class__.__name__,
            }
        except Exception as e:
            logger.error(f"Failed to get vector store info: {e}")
            return {"status": "error", "error": str(e)}