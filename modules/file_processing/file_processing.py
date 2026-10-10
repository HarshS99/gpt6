from typing import List, Dict, Any, Optional, Union, Tuple, cast
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
    Language,
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader,
)
from langchain_core.exceptions import LangChainException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileProcessingConfig(BaseModel):
    """Configuration for file processing parameters."""

    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    max_workers: int = Field(default=4, description="Maximum parallel workers for processing")
    supported_extensions: List[str] = Field(
        default=[
            ".pdf", ".docx", ".pptx", ".xlsx", ".txt",
            ".md", ".csv", ".json", ".html", ".htm"
        ],
        description="Supported file extensions"
    )

class FileProcessor:
    """
    A class to handle file processing for document-based question answering.
    Supports multiple file formats and provides document splitting capabilities.
    """

    def __init__(self, config: Optional[FileProcessingConfig] = None):
        """
        Initialize the FileProcessor with optional configuration.

        Args:
            config: FileProcessingConfig object or None to use defaults
        """
        self.config = config if config else FileProcessingConfig()
        self.text_splitter = self._create_text_splitter()

    def _create_text_splitter(self) -> TextSplitter:
        """Create a text splitter based on configuration."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def _get_loader_for_file(self, file_path: Path) -> Optional[Any]:
        """
        Get the appropriate document loader for a given file.

        Args:
            file_path: Path to the file

        Returns:
            Document loader instance or None if unsupported
        """
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return PyPDFLoader(str(file_path))
        elif extension == ".docx":
            return Docx2txtLoader(str(file_path))
        elif extension == ".pptx":
            return UnstructuredPowerPointLoader(str(file_path))
        elif extension in (".xlsx", ".csv"):
            return UnstructuredExcelLoader(str(file_path))
        elif extension == ".txt":
            return TextLoader(str(file_path))
        elif extension in (".md", ".html", ".htm"):
            # For markdown and HTML, use TextLoader with appropriate encoding
            return TextLoader(str(file_path), encoding="utf-8")
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return None

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if a file is supported based on its extension."""
        return file_path.suffix.lower() in self.config.supported_extensions

    def _process_single_file(self, file_path: Path) -> List[Document]:
        """
        Process a single file and return a list of documents.

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects

        Raises:
            LangChainException: If file processing fails
        """
        if not self._is_supported_file(file_path):
            raise LangChainException(f"File type not supported: {file_path}")

        try:
            loader = self._get_loader_for_file(file_path)
            if loader is None:
                raise LangChainException(f"No loader available for file: {file_path}")

            logger.info(f"Processing file: {file_path}")
            documents = loader.load()

            # Split documents into chunks
            split_documents = self.text_splitter.split_documents(documents)

            logger.info(f"Processed {file_path}: {len(split_documents)} chunks")
            return split_documents

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise LangChainException(f"Failed to process file {file_path}: {str(e)}")

    def process_files(
        self,
        file_paths: List[Union[str, Path]],
        parallel: bool = True
    ) -> List[Document]:
        """
        Process multiple files and return combined documents.

        Args:
            file_paths: List of file paths (strings or Path objects)
            parallel: Whether to process files in parallel

        Returns:
            List of Document objects from all processed files

        Raises:
            LangChainException: If any file processing fails
        """
        # Convert strings to Path objects
        paths = [Path(fp) if isinstance(fp, str) else fp for fp in file_paths]

        # Filter out unsupported files
        supported_paths = [p for p in paths if self._is_supported_file(p)]
        unsupported_paths = [p for p in paths if not self._is_supported_file(p)]

        if unsupported_paths:
            logger.warning(f"Skipping unsupported files: {unsupported_paths}")

        if not supported_paths:
            logger.warning("No supported files to process")
            return []

        # Process files
        documents = []

        if parallel:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._process_single_file, path): path
                    for path in supported_paths
                }

                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        result = future.result()
                        documents.extend(result)
                    except Exception as e:
                        logger.error(f"Error in parallel processing for {path}: {str(e)}")
                        continue
        else:
            for path in supported_paths:
                try:
                    result = self._process_single_file(path)
                    documents.extend(result)
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
                    continue

        logger.info(f"Total documents processed: {len(documents)}")
        return documents

    def process_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        parallel: bool = True
    ) -> List[Document]:
        """
        Process all supported files in a directory.

        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            parallel: Whether to process files in parallel

        Returns:
            List of Document objects from all processed files

        Raises:
            LangChainException: If directory doesn't exist or is empty
        """
        path = Path(directory_path)

        if not path.exists():
            raise LangChainException(f"Directory does not exist: {path}")

        if not path.is_dir():
            raise LangChainException(f"Path is not a directory: {path}")

        # Find all supported files
        if recursive:
            files = list(path.rglob("*"))
        else:
            files = list(path.glob("*"))

        supported_files = [f for f in files if f.is_file() and self._is_supported_file(f)]

        if not supported_files:
            logger.warning(f"No supported files found in directory: {path}")
            return []

        logger.info(f"Found {len(supported_files)} supported files in {path}")
        return self.process_files(supported_files, parallel=parallel)

    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if a file is supported and can be processed.

        Args:
            file_path: Path to the file

        Returns:
            True if file is valid and can be processed, False otherwise
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path

        if not path.exists():
            logger.error(f"File does not exist: {path}")
            return False

        if not path.is_file():
            logger.error(f"Path is not a file: {path}")
            return False

        if not self._is_supported_file(path):
            logger.error(f"Unsupported file type: {path.suffix}")
            return False

        return True


class DocumentProcessor:
    """Compatibility wrapper expected by the UI.

    Provides a `process_document(path: Path)` method that processes a single
    file and returns the list of document chunks.
    """

    def __init__(self, config: Optional[FileProcessingConfig] = None):
        self._processor = FileProcessor(config=config)

    def process_document(self, file_path: Union[str, Path]) -> List[Document]:
        path = Path(file_path) if isinstance(file_path, (str,)) else file_path
        if not self._processor.validate_file(path):
            return []
        return self._processor.process_files([path], parallel=False)