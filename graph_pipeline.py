import os
import itertools
from typing import List, Optional, TypedDict
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

# --- API Key Management ---
class ApiKeyManager:
    """
    Manages a pool of API keys for round-robin usage to distribute load.
    """

    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("API key list cannot be empty.")
        self.api_keys = itertools.cycle(api_keys)
        print(f"API Key Manager initialized with {len(api_keys)} keys.")

    def get_next_key(self) -> str:
        return next(self.api_keys)

    def get_llm(self, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
        """Provides a ChatGoogleGenerativeAI instance with the next API key."""
        api_key = self.get_next_key()
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=temperature,
        )


# Initialize the manager with keys from environment variables
api_keys_str = os.getenv("GOOGLE_API_KEYS", "")
api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
if not api_keys:
    print("Warning: GOOGLE_API_KEYS environment variable not set or empty. Using fallback.")
    # Add a fallback or raise an error if no keys are available
    # For this example, we'll assume it might be set elsewhere, but in production, you'd want to fail fast.

api_key_manager = ApiKeyManager(api_keys) if api_keys else None


# --- Document Loading (Expanded) ---
def load_and_split_documents(doc_path: str, password: Optional[str] = None):
    """
    Loads PDF, DOCX, or EML documents and splits them into chunks.
    """
    print(f"Loading document from path: {doc_path}...")
    _, file_extension = os.path.splitext(doc_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path=doc_path, password=password)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path=doc_path)
    elif file_extension == '.eml':
        loader = UnstructuredEmailLoader(file_path=doc_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")
    return chunks


# --- Vector Store and Retriever ---
def create_vector_store(chunks):
    """Creates an in-memory Chroma vector store."""
    print("Creating in-memory vector store...")
    # Note: Ensure the API key is available for the embedding model
    embedding_key = api_keys[0] if api_keys else os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=embedding_key)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("In-memory vector store created successfully.")
    return vector_store


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- LangGraph Definition ---

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The question to answer.
        context: The retrieved context from the document.
        answer: The generated answer.
        validation_result: The result of the validation step ("yes" or "no").
        retry_count: The number of times generation has been retried.
    """
    question: str
    context: List[str]
    answer: str
    validation_result: str
    retry_count: int


def create_rag_graph(chunks, vector_store) -> Runnable:
    """Builds and compiles the LangGraph for the RAG process with validation."""
    print("Creating RAG graph with validation and retry logic...")

    # 1. Define Retrievers
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6]  # Slightly more weight on semantic search
    )

    # 2. Define Graph Nodes
    def retrieve_node(state: GraphState):
        """Retrieves documents based on the question."""
        print("---NODE: RETRIEVE---")
        question = state["question"]
        documents = ensemble_retriever.invoke(question)
        return {"context": documents, "question": question}

    def generate_node(state: GraphState):
        """Generates an answer using the retrieved context."""
        print("---NODE: GENERATE---")
        question = state["question"]
        context = state["context"]

        generation_prompt = ChatPromptTemplate.from_template(
            """
            You are an AI assistant for interpreting insurance policies. Your answers must be accurate, concise, and strictly based on the provided context.
            - Limit responses to 49 words.
            - If the context does not contain the answer, respond with "Insufficient information."
            - Do not use introductory phrases like "The policy states..."
            - Translate complex terms into simple language.
            - Provide answers in a single, clear paragraph.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:
            """
        )

        llm = api_key_manager.get_llm()
        rag_chain = (
                generation_prompt
                | llm
                | StrOutputParser()
        )

        formatted_context = format_docs(context)
        answer = rag_chain.invoke({"context": formatted_context, "question": question})
        return {"answer": answer}

    def validate_node(state: GraphState):
        """Validates if the answer is supported by the context."""
        print("---NODE: VALIDATE---")
        question = state["question"]
        context = state["context"]
        answer = state["answer"]

        validation_prompt = ChatPromptTemplate.from_template(
            """
            Based ONLY on the provided CONTEXT, does the ANSWER correctly and factually respond to the QUESTION?
            Do not assess the answer's quality, only its factual grounding in the context.
            Respond with only 'yes' or 'no'.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:
            {answer}

            DECISION (yes/no):
            """
        )

        # Use a fast, low-temp LLM for this simple classification task
        validator_llm = api_key_manager.get_llm(temperature=0.0)

        validation_chain = (
                validation_prompt
                | validator_llm
                | StrOutputParser()
        )

        formatted_context = format_docs(context)
        result = validation_chain.invoke({
            "context": formatted_context,
            "question": question,
            "answer": answer
        })

        print(f"Validation result: {result.lower().strip()}")
        return {"validation_result": result.lower().strip()}

    # 3. Define Conditional Edge
    def should_continue(state: GraphState):
        """Determines whether to retry generation or end."""
        print("---EDGE: SHOULD_CONTINUE---")
        retry_count = state.get("retry_count", 0)
        if state["validation_result"] == "yes" or retry_count >= 1:  # Max 1 retry
            print("Validation successful or retries exceeded. Ending.")
            return "end"
        else:
            print("Validation failed. Retrying generation.")
            # Increment retry count before looping back
            state["retry_count"] = retry_count + 1
            return "continue"

    # 4. Build the Graph
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "continue": "generate",
            "end": END,
        },
    )

    # Compile the graph into a runnable object
    app = workflow.compile()
    print("RAG graph compiled successfully.")
    return app
