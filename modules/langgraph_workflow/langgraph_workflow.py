from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path
import logging
from langgraph.graph import Graph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from modules.ai_model.ai_model import AIModel
from modules.document_processor.document_processor import DocumentProcessor
from modules.memory_manager.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """
    Represents the state of the LangGraph workflow.

    Attributes:
        messages: List of chat messages exchanged in the conversation
        documents: List of relevant documents retrieved for the query
        question: The current user question being processed
        generation: The AI's response generation
        chat_history: Previous conversation history for context
    """
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    documents: Optional[List[Any]]
    question: Optional[str]
    generation: Optional[str]
    chat_history: Optional[List[Dict[str, Any]]]

class LangGraphWorkflow:
    """
    A LangGraph-based workflow for managing AI chatbot conversations with document retrieval.

    This class orchestrates the interaction between the AI model, document retriever,
    and memory management components using a graph-based workflow.
    """

    def __init__(
        self,
        ai_model: AIModel,
        document_processor: DocumentProcessor,
        memory_manager: MemoryManager,
        tools: Optional[List[BaseTool]] = None,
        max_history_length: int = 10
    ):
        """
        Initialize the LangGraph workflow.

        Args:
            ai_model: The AI model instance for generating responses
            document_processor: The document processing component for retrieval
            memory_manager: The memory management component
            tools: Optional list of tools that the AI can use
            max_history_length: Maximum number of previous messages to keep in history
        """
        self.ai_model = ai_model
        self.document_processor = document_processor
        self.memory_manager = memory_manager
        self.tools = tools or []
        self.max_history_length = max_history_length

        # Initialize the graph
        self.graph = Graph()

        # Configure memory for persistence
        self.memory = MemorySaver()

        # Build the workflow graph
        self._build_workflow()

    def _build_workflow(self) -> None:
        """
        Build the LangGraph workflow with nodes and edges.

        The workflow consists of:
        1. Message processing node
        2. Document retrieval node
        3. Response generation node
        4. Conditional edges for decision making
        """
        # Define nodes
        self.graph.add_node("process_message", self._process_message_node)
        self.graph.add_node("retrieve_documents", self._retrieve_documents_node)
        self.graph.add_node("generate_response", self._generate_response_node)
        self.graph.add_node("call_tool", self._call_tool_node)

        # Add tool nodes if tools are provided
        if self.tools:
            tool_node = ToolNode(tools=self.tools)
            self.graph.add_node("tools", tool_node)

        # Define edges
        self.graph.add_edge("process_message", "retrieve_documents")
        self.graph.add_edge("retrieve_documents", "generate_response")

        # Conditional edge from generate_response
        self.graph.add_conditional_edges(
            "generate_response",
            self._should_use_tools,
            {
                "tools": "call_tool",
                "generate": "generate_response"
            }
        )

        # Conditional edge from call_tool
        self.graph.add_conditional_edges(
            "call_tool",
            lambda state: "generate" if state.get("documents") else "generate_response",
            {
                "generate": "generate_response"
            }
        )

        # Set entry and finish points
        self.graph.set_entry_point("process_message")
        self.graph.set_finish_point("generate_response")

    def _process_message_node(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Process the incoming message and update the state.

        Args:
            state: Current graph state
            config: Runnable configuration

        Returns:
            Updated state with processed message
        """
        # Get the last message from the state
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return state

        # Update the question in state
        return {
            "question": last_message.content,
            "messages": messages
        }

    def _retrieve_documents_node(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Retrieve relevant documents based on the user's question.

        Args:
            state: Current graph state
            config: Runnable configuration

        Returns:
            Updated state with retrieved documents
        """
        question = state.get("question")
        if not question:
            return {"documents": []}

        try:
            # Retrieve documents using the document processor
            documents = self.document_processor.retrieve_documents(question)
            logger.info(f"Retrieved {len(documents)} documents for question: {question}")

            return {
                "documents": documents,
                "question": question
            }
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {"documents": []}

    def _generate_response_node(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Generate a response using the AI model with retrieved documents.

        Args:
            state: Current graph state
            config: Runnable configuration

        Returns:
            Updated state with generated response
        """
        question = state.get("question")
        documents = state.get("documents", [])
        messages = state.get("messages", [])

        if not question:
            return {"generation": "I didn't receive a valid question."}

        try:
            # Prepare context from documents
            context = "\n\n".join([doc.page_content for doc in documents]) if documents else "No documents found."

            # Prepare chat history for context
            chat_history = self.memory_manager.get_chat_history(
                max_length=self.max_history_length
            )

            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful AI assistant. Use the following context to answer the question."),
                *chat_history,
                HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")
            ])

            # Generate response
            chain = prompt | self.ai_model.model | StrOutputParser()
            response = chain.invoke({"context": context, "question": question}, config)

            logger.info(f"Generated response for question: {question}")

            return {
                "generation": response,
                "messages": messages + [HumanMessage(content=question), AIMessage(content=response)]
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = "Sorry, I encountered an error while generating a response."
            return {
                "generation": error_msg,
                "messages": messages + [HumanMessage(content=question), AIMessage(content=error_msg)]
            }

    def _call_tool_node(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Call a tool based on the generated response.

        Args:
            state: Current graph state
            config: Runnable configuration

        Returns:
            Updated state after tool execution
        """
        question = state.get("question")
        documents = state.get("documents", [])
        messages = state.get("messages", [])

        if not question:
            return {"generation": "I didn't receive a valid question for tool execution."}

        try:
            # Get the last AI message
            last_ai_message = next(
                (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
                None
            )

            if not last_ai_message:
                return {"generation": "No AI message found to process with tools."}

            # Execute the tool
            tool_result = self.tools[0].invoke({"question": question, "context": documents})

            logger.info(f"Tool executed successfully for question: {question}")

            return {
                "generation": tool_result,
                "messages": messages + [AIMessage(content=tool_result)]
            }
        except Exception as e:
            logger.error(f"Error calling tool: {e}")
            error_msg = f"Sorry, I encountered an error while using the tool: {str(e)}"
            return {
                "generation": error_msg,
                "messages": messages + [AIMessage(content=error_msg)]
            }

    def _should_use_tools(self, state: GraphState) -> str:
        """
        Determine whether to use tools based on the question and available tools.

        Args:
            state: Current graph state

        Returns:
            "tools" if tools should be used, "generate" otherwise
        """
        question = state.get("question", "")
        has_tools = bool(self.tools)
        has_documents = bool(state.get("documents", []))

        # Simple heuristic: use tools if we have tools available and the question is complex
        # In a real implementation, you might use more sophisticated logic
        if has_tools and ("calculate" in question.lower() or "search" in question.lower()):
            return "tools"

        return "generate"

    def invoke(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Invoke the LangGraph workflow with input data.

        Args:
            input_data: Input data for the workflow
            config: Optional runnable configuration

        Returns:
            The final state after workflow execution
        """
        if config is None:
            config = RunnableConfig()

        # Initialize state
        initial_state = {
            "messages": input_data.get("messages", []),
            "documents": None,
            "question": None,
            "generation": None,
            "chat_history": input_data.get("chat_history", [])
        }

        try:
            # Compile and invoke the graph
            app = self.graph.compile(checkpointer=self.memory)
            result = app.invoke(initial_state, config)

            # Update chat history in memory manager
            if result.get("messages"):
                self.memory_manager.add_to_chat_history(result["messages"])

            return result
        except Exception as e:
            logger.error(f"Error invoking LangGraph workflow: {e}")
            return {
                "generation": f"Workflow error: {str(e)}",
                "messages": input_data.get("messages", [])
            }

    def stream(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None):
        """
        Stream the workflow execution for real-time updates.

        Args:
            input_data: Input data for the workflow
            config: Optional runnable configuration

        Yields:
            Intermediate states during workflow execution
        """
        if config is None:
            config = RunnableConfig()

        # Initialize state
        initial_state = {
            "messages": input_data.get("messages", []),
            "documents": None,
            "question": None,
            "generation": None,
            "chat_history": input_data.get("chat_history", [])
        }

        try:
            # Compile and stream the graph
            app = self.graph.compile(checkpointer=self.memory)
            for chunk in app.stream(initial_state, config):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming LangGraph workflow: {e}")
            yield {
                "generation": f"Workflow error: {str(e)}",
                "messages": input_data.get("messages", [])
            }