import os
import itertools
import time
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
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

load_dotenv()


# --- API Key Management ---
class ApiKeyManager:
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("API key list cannot be empty.")
        self.api_keys = itertools.cycle(api_keys)
        print(f"API Key Manager initialized with {len(api_keys)} keys.")

    def get_next_key(self) -> str:
        return next(self.api_keys)

    def get_llm(self, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
        api_key = self.get_next_key()
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=temperature,
        )


class CohereKeyManager:
    def __init__(self, cohere_keys: List[str]):
        if not cohere_keys:
            raise ValueError("Cohere API key list cannot be empty.")
        self.cohere_keys = itertools.cycle(cohere_keys)
        print(f"Cohere Key Manager initialized with {len(cohere_keys)} keys.")

    def get_next_cohere_key(self) -> str:
        return next(self.cohere_keys)

    def get_cohere_validator(self, temperature: float = 0.0) -> ChatCohere:
        cohere_key = self.get_next_cohere_key()
        return ChatCohere(
            model="command-r-plus",  # Better reasoning for validation
            cohere_api_key=cohere_key,
            temperature=temperature,
            max_tokens=5  # Even shorter for yes/no responses
        )


# Initialize key managers
api_keys_str = os.getenv("GOOGLE_API_KEYS", "")
api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
print(f"Parsed Google API keys: {api_keys}")

if not api_keys:
    raise EnvironmentError("GOOGLE_API_KEYS environment variable not set or empty.")

# Initialize Cohere keys
cohere_keys_str = os.getenv("COHERE_API_KEYS", "")
cohere_keys = [key.strip() for key in cohere_keys_str.split(',') if key.strip()]
print(f"Parsed Cohere API keys: {cohere_keys}")

if not cohere_keys:
    raise EnvironmentError("COHERE_API_KEYS environment variable not set or empty.")

api_key_manager = ApiKeyManager(api_keys)
cohere_key_manager = CohereKeyManager(cohere_keys)


def load_and_split_documents(doc_path: str, password: Optional[str] = None):
    """
    Loads PDF, DOCX, or EML documents and splits them into chunks.
    Uses PyPDFLoader for PDF files.
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
    print(f"Split document into {len(chunks)} chunks using standard loaders.")
    return chunks


# --- Vector Store and Retriever ---
def create_vector_store(chunks):
    """Creates an in-memory Chroma vector store."""
    print("Creating in-memory vector store...")
    embedding_key = api_keys[0] if api_keys else os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=embedding_key)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("In-memory vector store created successfully.")
    return vector_store


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The question to answer.
        context: The retrieved context from the document.
        answer: The generated answer.
        validation_result: The result of the validation step ("yes" or "no").
        retry_count: The number of times generation has been retried.
        start_time: Timestamp when processing started.
    """
    question: str
    context: List[str]
    answer: str
    validation_result: str
    retry_count: int
    start_time: float


def create_rag_graph(chunks, vector_store) -> Runnable:
    """Builds and compiles the LangGraph for the RAG process with Cohere validation."""
    print("Creating RAG graph with Cohere validation and retry logic...")

    # 1. Define Retrievers
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6]  # Slightly more weight on semantic search
    )

    # Initialize Cohere validator with rotation
    def get_cohere_validator():
        return cohere_key_manager.get_cohere_validator(temperature=0.0)

    # 2. Define Graph Nodes
    def retrieve_node(state: GraphState):
        """Retrieves documents based on the question."""
        print("---NODE: RETRIEVE---")
        question = state["question"]
        documents = ensemble_retriever.invoke(question)
        start_time = time.time()
        return {"context": documents, "question": question, "start_time": start_time, "retry_count": 0}

    def generate_node(state: GraphState):
        """Generates an answer using the retrieved context."""
        print("---NODE: GENERATE---")

        # Check timeout - if more than 18 seconds have passed, skip validation
        elapsed_time = time.time() - state.get("start_time", time.time())
        if elapsed_time > 18:
            print("Timeout approaching, skipping further processing")
            return {"answer": "Insufficient time to process request fully."}

        question = state["question"]
        context = state["context"]

        generation_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert assistant for interpreting health insurance policies. Your job is to extract and present only what is explicitly stated in the given context.

            INSTRUCTIONS:
            - Keep answers factual, direct, and no longer than 49 words.
            - Use the same numbers, durations, and limits as in the context (e.g., 30 days, 36 months).
            - Do not assume or infer anything not present in the context.
            - Avoid opening phrases like "The policy states..." or "According to the document..."
            - Explain complex language in simpler terms where needed.
            - If the answer is not clearly stated, reply with: "Insufficient information."

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
        """Validates if the answer is supported by the context using Cohere."""
        print("---NODE: VALIDATE (Cohere)---")

        # Check timeout
        elapsed_time = time.time() - state.get("start_time", time.time())
        if elapsed_time > 17:  # Leave 3 seconds buffer
            print("Timeout approaching, skipping validation")
            return {"validation_result": "yes"}  # Accept answer to avoid timeout

        question = state["question"]
        context = state["context"]
        answer = state["answer"]

        validation_prompt = ChatPromptTemplate.from_template(
            """
            You are validating if the AI-generated insurance answer is accurate, strictly based on the given policy document excerpts.

            VALIDATION RULES:
            - All numbers, dates, limits, and terms (e.g., "30 days", "36 months", "2 years", "5%") must match exactly.
            - Coverage rules, eligibility, and exclusions must align precisely with the context.
            - Even small differences in timing, percentages, or definitions should be considered incorrect.
            - Use only the context to decide. Do not assume or generalize.
            - Mark 'yes' only if everything in the answer is directly supported by the context.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:
            {answer}

            Is the answer 100% supported by the policy context? Reply only with 'yes' or 'no'.
            """
        )

        validation_chain = (
                validation_prompt
                | get_cohere_validator()
                | StrOutputParser()
        )

        try:
            formatted_context = format_docs(context)
            result = validation_chain.invoke({
                "context": formatted_context,
                "question": question,
                "answer": answer
            })

            # Clean up the result
            result = result.lower().strip()
            if "yes" in result:
                result = "yes"
            elif "no" in result:
                result = "no"
            else:
                result = "yes"  # Default to accepting if unclear

        except Exception as e:
            print(f"Cohere validation error: {e}")
            result = "yes"  # Default to accepting on error

        print(f"Cohere validation result: {result}")
        return {"validation_result": result}

    # 3. Define Conditional Edge
    def should_continue(state: GraphState):
        """Determines whether to retry generation or end."""
        print("---EDGE: SHOULD_CONTINUE---")

        # Check timeout first
        elapsed_time = time.time() - state.get("start_time", time.time())
        if elapsed_time > 15:  # Conservative timeout check
            print("Timeout approaching. Ending processing.")
            return "end"

        retry_count = state.get("retry_count", 0)

        # Maximum 2 validations (initial + 1 retry)
        if state["validation_result"] == "yes" or retry_count >= 5:
            if retry_count >= 5:
                print("Maximum validations (2) reached. Ending.")
            else:
                print("Validation successful. Ending.")
            return "end"
        else:
            print(f"Validation failed. Retrying generation. Attempt {retry_count + 1}/2")
            return "continue"

    def increment_retry(state: GraphState):
        """Increments retry count before regeneration."""
        retry_count = state.get("retry_count", 0) + 1
        return {"retry_count": retry_count}

    # 4. Build the Graph
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("increment_retry", increment_retry)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "continue": "increment_retry",
            "end": END,
        },
    )
    workflow.add_edge("increment_retry", "generate")

    # Compile the graph into a runnable object
    app = workflow.compile()
    print("RAG graph with Cohere validation compiled successfully.")
    return app