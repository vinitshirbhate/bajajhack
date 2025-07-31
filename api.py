import os
import requests
import tempfile
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Dict
import uvicorn
from langchain_core.runnables import Runnable
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from rag_pipeline import (
    create_rag_chain,
    load_and_split_documents,
    create_vector_store,
)

load_dotenv()
app = FastAPI(title="High-Performance Q&A API", version="4.0.0")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAG_CHAIN_CACHE: Dict[str, Runnable] = {}


class ApiQueryRequest(BaseModel):
    document_url: str = Field(..., alias="documents")
    questions: List[str]


class ApiResponse(BaseModel):
    answers: List[str]


def get_or_create_rag_chain(doc_url: str) -> Runnable:
    """Checks cache for a RAG chain; creates and caches it if not found."""
    if doc_url in RAG_CHAIN_CACHE:
        print(f"RAG chain for {doc_url} found in cache.")
        return RAG_CHAIN_CACHE[doc_url]

    print(f"RAG chain for {doc_url} not in cache. Creating new chain...")
    temp_file_path = None
    try:
        response = requests.get(doc_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        chunks = load_and_split_documents(temp_file_path)
        vector_store = create_vector_store(chunks)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"),
                                     temperature=0)

        rag_chain = create_rag_chain(chunks, vector_store, llm)
        RAG_CHAIN_CACHE[doc_url] = rag_chain
        return rag_chain
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/api/v2/hackrx/run", response_model=ApiResponse)
async def ask_question(request: ApiQueryRequest):
    """Processes questions concurrently using a cached or newly created RAG chain."""
    try:
        rag_chain = get_or_create_rag_chain(request.document_url)

        tasks = [rag_chain.ainvoke(q) for q in request.questions]

        answers_list = await asyncio.gather(*tasks)

        return ApiResponse(answers=answers_list)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # default for local dev
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)