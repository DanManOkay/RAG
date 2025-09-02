from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import shutil
from pathlib import Path
import asyncio
import logging

# Import our RAG system
from advanced_rag import RAGSystem, RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Advanced RAG API",
    description="High-quality Retrieval-Augmented Generation API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_system: Optional[RAGSystem] = None
system_ready = False

# Pydantic models
class RAGConfigRequest(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    temperature: float = 0.1
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4-turbo-preview"
    collection_name: str = "documents"

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]
    query: str
    processing_time: float

class SystemStatus(BaseModel):
    ready: bool
    message: str
    config: Optional[Dict[str, Any]] = None

class DocumentUploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_chunks: int

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Advanced RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/status"
    }

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    global system_ready, rag_system
    
    if system_ready and rag_system:
        return SystemStatus(
            ready=True,
            message="System is ready",
            config={
                "chunk_size": rag_system.config.chunk_size,
                "top_k": rag_system.config.top_k,
                "temperature": rag_system.config.temperature,
                "llm_model": rag_system.config.llm_model,
                "embedding_model": rag_system.config.embedding_model
            }
        )
    else:
        return SystemStatus(
            ready=False,
            message="System not initialized"
        )

@app.post("/initialize", response_model=SystemStatus)
async def initialize_system(
    config_request: RAGConfigRequest,
    background_tasks: BackgroundTasks
):
    """Initialize the RAG system"""
    global rag_system, system_ready
    
    try:
        # Create config
        config = RAGConfig(
            chunk_size=config_request.chunk_size,
            chunk_overlap=config_request.chunk_overlap,
            top_k=config_request.top_k,
            temperature=config_request.temperature,
            embedding_model=config_request.embedding_model,
            llm_model=config_request.llm_model,
            collection_name=config_request.collection_name
        )
        
        # Initialize system
        rag_system = RAGSystem(config)
        
        # Setup system in background if documents exist
        documents_path = "./documents"
        if os.path.exists(documents_path) and os.listdir(documents_path):
            background_tasks.add_task(setup_system_background, documents_path)
            message = "System initialized. Document processing started in background."
        else:
            message = "System initialized. Upload documents to complete setup."
        
        return SystemStatus(
            ready=False,  # Will be True once background setup completes
            message=message,
            config=config_request.dict()
        )
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

async def setup_system_background(documents_path: str):
    """Background task to setup the system"""
    global system_ready
    
    try:
        logger.info("Starting background system setup...")
        rag_system.setup_system(documents_path)
        system_ready = True
        logger.info("Background system setup completed!")
    except Exception as e:
        logger.error(f"Background setup failed: {str(e)}")
        system_ready = False

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload documents to the system"""
    global rag_system, system_ready
    
    if not rag_system:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        # Create documents directory
        documents_dir = Path("./documents")
        documents_dir.mkdir(exist_ok=True)
        
        uploaded_files = []
        
        # Save uploaded files
        for file in files:
            if file.filename:
                file_path = documents_dir / file.filename
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                uploaded_files.append(file.filename)
                logger.info(f"Uploaded file: {file.filename}")
        
        # Reprocess documents in background
        background_tasks.add_task(reprocess_documents)
        
        return DocumentUploadResponse(
            message=f"Uploaded {len(uploaded_files)} files. Processing started.",
            files_processed=uploaded_files,
            total_chunks=0  # Will be updated after processing
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def reprocess_documents():
    """Background task to reprocess documents after upload"""
    global system_ready
    
    try:
        logger.info("Reprocessing documents...")
        system_ready = False
        rag_system.setup_system("./documents")
        system_ready = True
        logger.info("Document reprocessing completed!")
    except Exception as e:
        logger.error(f"Reprocessing failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_system(query_request: QueryRequest):
    """Query the RAG system"""
    global rag_system, system_ready
    
    if not system_ready or not rag_system:
        raise HTTPException(
            status_code=400, 
            detail="System not ready. Please initialize and upload documents first."
        )
    
    try:
        import time
        start_time = time.time()
        
        # Override top_k if provided
        if query_request.top_k:
            original_top_k = rag_system.config.top_k
            rag_system.config.top_k = query_request.top_k
        
        # Get response
        response = rag_system.chat(query_request.question)
        
        # Restore original top_k
        if query_request.top_k:
            rag_system.config.top_k = original_top_k
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=response["answer"],
            source_documents=response["source_documents"],
            query=response["query"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/evaluate")
async def evaluate_system(test_questions: List[str]):
    """Evaluate the RAG system performance"""
    global rag_system, system_ready
    
    if not system_ready or not rag_system:
        raise HTTPException(status_code=400, detail="System not ready")
    
    try:
        results = rag_system.evaluate(test_questions)
        return {
            "evaluation_results": results,
            "test_questions": test_questions,
            "message": "Evaluation completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List uploaded documents"""
    documents_dir = Path("./documents")
    
    if not documents_dir.exists():
        return {"documents": [], "message": "No documents directory found"}
    
    documents = []
    for file_path in documents_dir.iterdir():
        if file_path.is_file():
            documents.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime
            })
    
    return {
        "documents": documents,
        "count": len(documents)
    }

@app.delete("/documents/{filename}")
async def delete_document(filename: str, background_tasks: BackgroundTasks):
    """Delete a document and reprocess"""
    documents_dir = Path("./documents")
    file_path = documents_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        file_path.unlink()
        
        # Reprocess remaining documents
        if rag_system:
            background_tasks.add_task(reprocess_documents)
        
        return {"message": f"Document {filename} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )