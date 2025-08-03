import os
import requests
import tempfile
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Dict
from langchain_core.runnables import Runnable
from fastapi.middleware.cors import CORSMiddleware

# Import the new graph creation logic
from graph_pipeline import (
    create_rag_graph,
    load_and_split_documents,
    create_vector_store,
    api_key_manager  # Import the manager to ensure it's initialized
)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="High-Performance Q&A API with Validation",
    version="5.0.0",
    description="An API that answers questions about documents using a self-correcting RAG graph."
)

# CORS configuration
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for compiled RAG graphs to avoid rebuilding for the same document
RAG_GRAPH_CACHE: Dict[str, Runnable] = {}
MAX_CACHE_SIZE = 10  # To prevent unbounded memory usage


class ApiQueryRequest(BaseModel):
    document_url: str = Field(..., alias="documents")
    questions: List[str]


class ApiResponse(BaseModel):
    answers: List[str]


def get_or_create_rag_graph(doc_url: str) -> Runnable:
    """
    Checks cache for a RAG graph; creates and caches it if not found.
    Handles document download and processing.
    """
    if doc_url in RAG_GRAPH_CACHE:
        print(f"RAG graph for {doc_url} found in cache.")
        return RAG_GRAPH_CACHE[doc_url]

    if not api_key_manager:
        raise HTTPException(status_code=500,
                            detail="API Key Manager is not configured. Check GOOGLE_API_KEYS environment variable.")
    print(f"RAG graph for {doc_url} not in cache. Creating new graph...")

    # Manage cache size
    if len(RAG_GRAPH_CACHE) >= MAX_CACHE_SIZE:
        # Simple FIFO cache eviction
        oldest_key = next(iter(RAG_GRAPH_CACHE))
        del RAG_GRAPH_CACHE[oldest_key]
        print(f"Cache full. Evicted graph for {oldest_key}")

    temp_file_path = None
    try:
        # Download the document
        response = requests.get(doc_url)
        response.raise_for_status()

        # Save to a temporary file with the correct extension
        suffix = os.path.splitext(doc_url.split('?')[0])[-1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # Process the document and build the graph
        chunks = load_and_split_documents(temp_file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract content from the document.")

        vector_store = create_vector_store(chunks)
        rag_graph = create_rag_graph(chunks, vector_store)

        # Cache the compiled graph
        RAG_GRAPH_CACHE[doc_url] = rag_graph
        return rag_graph
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/v1/hackrx/run", response_model=ApiResponse)
async def ask_question(request: ApiQueryRequest):
    """
    Processes questions concurrently using a cached or newly created RAG graph.
    """
    try:
        rag_graph = get_or_create_rag_graph(request.document_url)

        # Create a list of tasks to run the graph for each question
        tasks = []
        for q in request.questions:
            # Each question runs through the graph independently
            task_input = {"question": q, "retry_count": 0}
            tasks.append(rag_graph.ainvoke(task_input))

        # Gather results from all concurrent graph runs
        results = await asyncio.gather(*tasks)

        # Extract the final answer from each graph's state
        answers_list = [res.get("answer", "Error: No answer generated.") for res in results]

        return ApiResponse(answers=answers_list)

    except HTTPException as e:
        # Re-raise HTTP exceptions to let FastAPI handle them
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Provide a more generic error to the client for security
        raise HTTPException(status_code=500, detail="An internal server error occurred.")



