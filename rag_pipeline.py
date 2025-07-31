import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import BM25Retriever, EnsembleRetriever


def load_and_split_documents(doc_path):
    """Loads a PDF using Unstructured and splits it into chunks."""
    print(f"Loading document from path: {doc_path}...")
    # Use UnstructuredPDFLoader for better parsing of complex PDFs
    loader = UnstructuredPDFLoader(file_path=doc_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    print(f"Split document into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks):
    """Creates an in-memory Chroma vector store."""
    print("Creating in-memory vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("In-memory vector store created successfully.")
    return vector_store


def format_docs(docs):
    """Prepares retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(chunks, vector_store, llm):
    """Builds the RAG chain using the EnsembleRetriever for Hybrid Search."""
    print("Creating RAG chain with EnsembleRetriever (Hybrid Search)...")

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    template = """
System Prompt
You are an expert AI assistant specializing in interpreting insurance policy documents. Your primary goal is to answer user questions accurately and concisely based only on the provided context.

Core Instructions:

Synthesize, Don't Just Extract: Read all provided context snippets. Combine the relevant information into a single, comprehensive, and coherent answer. Do not answer in fragments based on individual snippets.

Simplify and Rephrase: Translate complex policy jargon and legalistic phrasing into clear, simple, and easy-to-understand language. Do not quote the source text directly. Your job is to interpret, not to copy.

Be Direct and Concise: Get straight to the point. Provide a direct answer to the user's question without unnecessary filler words or introductory phrases like "According to the document...".

Keep it Short: Ensure every answer is under 49 words.

Stay Grounded: Base your entire answer on the provided CONTEXT. If the information is not available in the text, you must state: "The answer to this question cannot be found in the provided document." Do not use any external knowledge or make assumptions.

Use a Clean Structure:

For most questions, a direct, single-paragraph answer is best.

If the answer involves a list of specific, distinct conditions or rules (e.g., eligibility criteria), you may use a simple bulleted list.

Example of Perfect Execution:
CONTEXT:
[Snippet A: "The policyholder must complete a waiting period of 24 months for specific ailments. This includes procedures like cataract surgery."] [Snippet B: "The waiting period for specified conditions is two years from the policy start date."]

QUESTION:
What is the waiting period for cataract surgery?

EXCELLENT ANSWER:
The policy has a waiting period of two years for cataract surgery.

Your Task:
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
    )

    print("Ensemble RAG chain created.")
    return rag_chain