import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file or similar
load_dotenv()

def get_llm_chain():
    # Initialize the LLM (using Groq as per the API key in .env.example)
    # Ensure GROQ_API_KEY is in the environment
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and friendly AI assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    # Return the runnable chain
    return prompt | llm
