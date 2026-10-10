from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.utils import guardrails
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseCache
from langchain_core.stores import InMemoryStore
from langchain_core.loaders import BaseLoader
from langchain_core.document_loaders import TextLoader
from langchain_core.document_loaders import PyPDFLoader
from langchain_core.document_loaders import Docx2txtLoader
from langchain_core.document_loaders import UnstructuredPowerPointLoader
from langchain_core.document_loaders import CSVLoader
from langchain_core.document_loaders import JSONLoader
from langchain_core.stores import BaseStore
from langchain_core.stores import LocalFileStore
from langchain_core.stores import UpstashRedisStore
from langchain_core.stores import RedisStore
from langchain_core.stores import MongoDBStore
from langchain_core.stores import PostgresStore
from langlin import ChatLlama

logger = logging.getLogger(__name__)

class AIModel:
    """
    A comprehensive AI model handler that manages document processing, vector storage,
    and conversation with retrieval-augmented generation capabilities.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: str = "./chroma_db",
        device: str = "cpu",
        **kwargs: Any
    ):
        """
        Initialize the AI model with configuration parameters.

        Args:
            model_name: Name of the LLM model to use
            embedding_model_name: Name of the embedding model
            temperature: Temperature for text generation
            max_tokens: Maximum tokens to generate
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between text chunks
            persist_directory: Directory to persist vector database
            device: Device to run models on ('cpu' or 'cuda')
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = Path(persist_directory)
        self.device = device
        self.kwargs = kwargs

        # Initialize components
        self.embedding_model: Optional[Embeddings] = None
        self.vector_store: Optional[VectorStore] = None
        self.retriever: Optional[BaseRetriever] = None
        self.llm: Optional[BaseChatModel] = None
        self.prompt_template: Optional[ChatPromptTemplate] = None

        # Initialize the model components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all model components."""
        try:
            # Initialize embedding model
            self._initialize_embedding_model()

            # Initialize vector store
            self._initialize_vector_store()

            # Initialize LLM
            self._initialize_llm()

            # Initialize prompt template
            self._initialize_prompt_template()

            logger.info("AIModel components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AIModel components: {str(e)}")
            raise

    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model."""
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info(f"Embedding model {self.embedding_model_name} initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def _initialize_vector_store(self) -> None:
        """Initialize the vector store."""
        try:
            # Create persist directory if it doesn't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Initialize Chroma vector store
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embedding_model
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            logger.info(f"Vector store initialized at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def _initialize_llm(self) -> None:
        """Initialize the language model."""
        try:
            # For this example, we'll use a local model
            # In production, you might use a hosted model or API
            self.llm = ChatLlama(
                model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                n_ctx=self.max_tokens,
                n_threads=4,
                n_gpu_layers=32 if self.device == "cuda" else 0,
                temperature=self.temperature,
                **self.kwargs
            )
            logger.info(f"LLM {self.model_name} initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def _initialize_prompt_template(self) -> None:
        """Initialize the prompt template for the chat model."""
        try:
            # System message template
            system_message_prompt = SystemMessagePromptTemplate.from_template(
                """You are a helpful, creative, and very friendly AI assistant. "
                "You provide detailed and accurate responses to "
                "user questions. If you don't know the answer, say you don't know.

                Context:
                {context}

                Conversation history:
                {history}"""
            )

            # Human message template
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                "{question}"
            )

            # Chat prompt template
            self.prompt_template = ChatPromptTemplate.from_messages([
                system_message_prompt,
                MessagesPlaceholder(variable_name="history"),
                human_message_prompt
            ])

            logger.info("Prompt template initialized")
        except Exception as e:
            logger.error(f"Failed to initialize prompt template: {str(e)}")
            raise

    def _format_context(self, documents: List[Document]) -> str:
        """
        Format the retrieved documents into a readable context string.

        Args:
            documents: List of document chunks

        Returns:
            Formatted context string
        """
        context = []
        for doc in documents:
            context.append(f"Source: {doc.metadata.get('source', 'unknown')}\n")
            context.append(f"Content: {doc.page_content}\n\n")
        return "\n".join(context)

    def _format_history(self, history: List[Union[HumanMessage, AIMessage]]) -> str:
        """
        Format conversation history into a readable string.

        Args:
            history: List of message objects

        Returns:
            Formatted history string
        """
        formatted_history = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"Assistant: {msg.content}")
        return "\n".join(formatted_history)

    def _get_retrieval_augmented_chain(self) -> RunnableLambda:
        """
        Create a retrieval-augmented generation chain.

        Returns:
            Runnable chain for processing queries with retrieval
        """
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        retrieval_chain = RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"])
        ).assign(
            history=lambda x: self._format_history(x["history"])
        ) | self.prompt_template | self.llm | StrOutputParser()

        return retrieval_chain

    def ingest_documents(
        self,
        documents: List[Union[str, Path, Document]],
        document_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ingest documents into the vector store.

        Args:
            documents: List of document paths or Document objects
            document_type: Type of document ('text', 'pdf', 'docx', 'pptx', 'csv', 'json')
            metadata: Additional metadata for documents
        """
        try:
            # Load documents based on type
            loaded_docs = []
            for doc in documents:
                if isinstance(doc, (str, Path)):
                    if document_type == "text":
                        loader = TextLoader(str(doc))
                    elif document_type == "pdf":
                        loader = PyPDFLoader(str(doc))
                    elif document_type == "docx":
                        loader = Docx2txtLoader(str(doc))
                    elif document_type == "pptx":
                        loader = UnstructuredPowerPointLoader(str(doc))
                    elif document_type == "csv":
                        loader = CSVLoader(str(doc))
                    elif document_type == "json":
                        loader = JSONLoader(str(doc))
                    else:
                        raise ValueError(f"Unsupported document type: {document_type}")

                    loaded_docs.extend(loader.load())
                elif isinstance(doc, Document):
                    loaded_docs.append(doc)
                else:
                    raise ValueError(f"Unsupported document type: {type(doc)}")

            # Add metadata to documents
            if metadata:
                for doc in loaded_docs:
                    doc.metadata.update(metadata)

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            split_docs = text_splitter.split_documents(loaded_docs)

            # Add documents to vector store
            self.vector_store.add_documents(split_docs)
            self.vector_store.persist()

            logger.info(f"Ingested {len(split_docs)} document chunks")
        except Exception as e:
            logger.error(f"Failed to ingest documents: {str(e)}")
            raise

    def chat(
        self,
        message: str,
        history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
        retrieval: bool = True
    ) -> str:
        """
        Generate a response to a user message.

        Args:
            message: User message
            history: Conversation history
            retrieval: Whether to use retrieval-augmented generation

        Returns:
            Generated response
        """
        try:
            if history is None:
                history = []

            # Add user message to history
            history.append(HumanMessage(content=message))

            if retrieval:
                # Use retrieval-augmented generation
                context_docs = self.retriever.invoke(message)
                context = self._format_context(context_docs)

                # Create input for the chain
                input_data = {
                    "context": context,
                    "question": message,
                    "history": history[:-1]  # Exclude the current message
                }

                # Generate response
                response = self._get_retrieval_augmented_chain().invoke(input_data)
            else:
                # Use simple chat without retrieval
                response = self.llm.invoke(
                    self.prompt_template.format_prompt(
                        question=message,
                        history=self._format_history(history[:-1])
                    ).to_messages()
                ).content

            # Add AI response to history
            history.append(AIMessage(content=response))

            return response
        except Exception as e:
            logger.error(f"Failed to generate chat response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request."

    async def achat(
        self,
        message: str,
        history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
        retrieval: bool = True,
        config: Optional[RunnableConfig] = None
    ) -> str:
        """
        Asynchronously generate a response to a user message.

        Args:
            message: User message
            history: Conversation history
            retrieval: Whether to use retrieval-augmented generation
            config: Runnable configuration

        Returns:
            Generated response
        """
        try:
            if history is None:
                history = []

            # Add user message to history
            history.append(HumanMessage(content=message))

            if retrieval:
                # Use retrieval-augmented generation
                context_docs = await self.retriever.ainvoke(message)
                context = self._format_context(context_docs)

                # Create input for the chain
                input_data = {
                    "context": context,
                    "question": message,
                    "history": history[:-1]  # Exclude the current message
                }

                # Generate response
                response = await self._get_retrieval_augmented_chain().ainvoke(
                    input_data,
                    config=config
                )
            else:
                # Use simple chat without retrieval
                response = (await self.llm.ainvoke(
                    self.prompt_template.format_prompt(
                        question=message,
                        history=self._format_history(history[:-1])
                    ).to_messages(),
                    config=config
                )).content

            # Add AI response to history
            history.append(AIMessage(content=response))

            return response
        except Exception as e:
            logger.error(f"Failed to generate async chat response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request."

    def clear_vector_store(self) -> None:
        """Clear all documents from the vector store."""
        try:
            self.vector_store._collection.delete()
            self.vector_store.persist()
            logger.info("Vector store cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {str(e)}")
            raise

    def get_vector_store_size(self) -> int:
        """
        Get the number of documents in the vector store.

        Returns:
            Number of documents
        """
        try:
            return self.vector_store._collection.count()
        except Exception as e:
            logger.error(f"Failed to get vector store size: {str(e)}")
            raise

    def save_vector_store(self) -> None:
        """Persist the vector store to disk."""
        try:
            self.vector_store.persist()
            logger.info("Vector store persisted successfully")
        except Exception as e:
            logger.error(f"Failed to persist vector store: {str(e)}")
            raise

    def load_vector_store(self) -> None:
        """Load the vector store from disk."""
        try:
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embedding_model
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise