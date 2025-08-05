import sys
import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn

# Adjust the path to include the 'src' directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_pipeline import RAGPipeline

app = FastAPI(
    title="Financial Report RAG API",
    description="An API to query financial documents using a RAG pipeline.",
    version="1.0.0"
)

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_reports')
INDEX_FILE = "api_faiss_index.bin"
CHUNKS_FILE = "api_text_chunks.pkl"

# --- Globals ---
pipeline = RAGPipeline(
    data_directory=DATA_DIR, 
    index_path=INDEX_FILE, 
    chunks_path=CHUNKS_FILE
)
is_indexing = False

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

class SetupResponse(BaseModel):
    message: str

# --- Helper Functions ---
def run_indexing():
    global is_indexing
    is_indexing = True
    print("Background task: Starting indexing...")
    try:
        pipeline.setup_and_index()
        print("Background task: Indexing complete.")
    except Exception as e:
        print(f"Background task: An error occurred during indexing: {e}")
    finally:
        is_indexing = False

# --- API Endpoints ---
@app.get("/", summary="Health Check")
def read_root():
    """A simple health check endpoint."""
    return {"status": "API is running"}

@app.post("/setup", response_model=SetupResponse, summary="Setup and Index Documents")
def setup_documents(background_tasks: BackgroundTasks):
    """
    Triggers the document indexing process. This runs in the background.
    """
    global is_indexing
    if is_indexing:
        raise HTTPException(status_code=409, detail="Indexing is already in progress.")
    
    if os.path.exists(INDEX_FILE):
        return {"message": "Index already exists. No action taken. To re-index, please delete the index files first."}

    background_tasks.add_task(run_indexing)
    return {"message": "Document indexing has been started in the background."}

@app.post("/query", response_model=QueryResponse, summary="Query the Documents")
def query_documents(request: QueryRequest):
    """
    Asks a question to the indexed financial documents.
    """
    if is_indexing:
        raise HTTPException(status_code=503, detail="Service unavailable: Indexing is in progress. Please try again later.")
        
    if not os.path.exists(INDEX_FILE):
        raise HTTPException(status_code=400, detail="Index not found. Please run the /setup endpoint first.")

    try:
        answer = pipeline.execute_query(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {e}")

if __name__ == "__main__":
    print("--- IMPORTANT ---")
    print(f"Please make sure your financial report PDF (e.g., 'TSLA-Q1-2024-Update.pdf')")
    print(f"is placed inside the '{DATA_DIR}' directory before running.")
    print(f"-----------------")
    uvicorn.run(app, host="0.0.0.0", port=8000)
