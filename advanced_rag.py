import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

# Core imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Advanced features
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import tiktoken
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

# Utilities
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from rich.console import Console
from rich.progress import Progress

console = Console()
load_dotenv()

@dataclass
class RAGConfig:
    """Configuration for the RAG system"""
    vector_db_url: str = "http://localhost:6333"
    collection_name: str = "documents"
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4-turbo-preview"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    temperature: float = 0.1

class AdvancedDocumentProcessor:
    """Advanced document processing with multiple strategies"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
    def load_documents(self, directory_path: str) -> List[Dict]:
        """Load documents from directory with advanced processing"""
        console.print(f"[blue]Loading documents from {directory_path}[/blue]")
        
        loader = DirectoryLoader(
            directory_path,
            glob="**/*",
            loader_cls=UnstructuredFileLoader,
            loader_kwargs={"mode": "elements"}
        )
        
        documents = loader.load()
        
        # Process documents with metadata enhancement
        processed_docs = []
        for doc in documents:
            # Add enhanced metadata
            doc.metadata.update({
                'source_type': Path(doc.metadata['source']).suffix,
                'chunk_id': len(processed_docs),
                'processing_timestamp': pd.Timestamp.now().isoformat()
            })
            processed_docs.append(doc)
            
        return processed_docs
    
    def smart_chunk(self, documents: List[Dict]) -> List[Dict]:
        """Intelligent chunking based on document structure"""
        chunked_docs = []
        
        with Progress() as progress:
            task = progress.add_task("[green]Chunking documents...", total=len(documents))
            
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                
                # Add chunk-specific metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk.page_content)
                    })
                    chunked_docs.append(chunk)
                
                progress.advance(task)
        
        console.print(f"[green]Created {len(chunked_docs)} chunks from {len(documents)} documents[/green]")
        return chunked_docs

class HybridVectorStore:
    """Advanced vector store with hybrid search capabilities"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = QdrantClient(url=config.vector_db_url)
        self.embeddings = OpenAIEmbeddings(model=config.embedding_model)
        
        # Initialize collection
        self._initialize_collection()
        
    def _initialize_collection(self):
        """Initialize Qdrant collection with proper configuration"""
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.config.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=3072,  # OpenAI text-embedding-3-large dimension
                    distance=Distance.COSINE
                )
            )
            console.print(f"[green]Created collection: {self.config.collection_name}[/green]")
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to vector store with batch processing"""
        console.print("[blue]Adding documents to vector store...[/blue]")
        
        vector_store = Qdrant(
            client=self.client,
            collection_name=self.config.collection_name,
            embeddings=self.embeddings
        )
        
        # Batch process for efficiency
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
            console.print(f"[green]Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}[/green]")
    
    def similarity_search(self, query: str, k: int = None) -> List[Dict]:
        """Perform similarity search"""
        k = k or self.config.top_k
        
        vector_store = Qdrant(
            client=self.client,
            collection_name=self.config.collection_name,
            embeddings=self.embeddings
        )
        
        return vector_store.similarity_search(query, k=k)

class AdvancedRAGChain:
    """Advanced RAG chain with conversation memory and evaluation"""
    
    def __init__(self, config: RAGConfig, vector_store: HybridVectorStore):
        self.config = config
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature
        )
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Keep last 5 exchanges
        )
        
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup the RAG chain with custom prompt"""
        
        prompt_template = """
        You are an advanced AI assistant with access to a comprehensive document database. 
        Use the following context to answer the question accurately and comprehensively.
        
        Context from documents:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Instructions:
        1. Provide accurate, detailed answers based on the context
        2. If information is not in the context, clearly state this
        3. Reference specific documents when possible
        4. Maintain conversation continuity using chat history
        5. Be concise but thorough
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create retriever
        qdrant_store = Qdrant(
            client=self.vector_store.client,
            collection_name=self.config.collection_name,
            embeddings=self.vector_store.embeddings
        )
        
        retriever = qdrant_store.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        
        # Create the chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            memory=self.memory,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        console.print(f"[blue]Processing query: {question}[/blue]")
        
        # Get response - ADD chat_history here
        response = self.chain.invoke({
            "query": question, 
            "chat_history": []  # Add this line
        })

class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, rag_chain: AdvancedRAGChain):
        self.rag_chain = rag_chain
    
    def evaluate_responses(self, test_questions: List[str], ground_truths: List[str] = None):
        """Evaluate RAG responses using RAGAS metrics"""
        console.print("[blue]Evaluating RAG system...[/blue]")
        
        responses = []
        contexts = []
        
        for question in test_questions:
            result = self.rag_chain.query(question)
            responses.append(result["answer"])
            
            # Get contexts
            source_contexts = [doc["content"] for doc in result["source_documents"]]
            contexts.append(source_contexts)
        
        # Create dataset for evaluation
        eval_dataset = Dataset.from_dict({
            "question": test_questions,
            "answer": responses,
            "contexts": contexts,
            "ground_truths": ground_truths if ground_truths else [""] * len(test_questions)
        })
        
        # Evaluate
        metrics = [faithfulness, answer_relevancy, context_precision]
        results = evaluate(eval_dataset, metrics=metrics)
        
        console.print("[green]Evaluation completed![/green]")
        return results

class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.processor = AdvancedDocumentProcessor(self.config)
        self.vector_store = HybridVectorStore(self.config)
        self.rag_chain = None
        self.evaluator = None
    
    def setup_system(self, documents_path: str):
        """Setup the complete RAG system"""
        console.print("[bold blue]Setting up Advanced RAG System...[/bold blue]")
        
        # Process documents
        documents = self.processor.load_documents(documents_path)
        chunks = self.processor.smart_chunk(documents)
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        # Setup RAG chain
        self.rag_chain = AdvancedRAGChain(self.config, self.vector_store)
        self.evaluator = RAGEvaluator(self.rag_chain)
        
        console.print("[bold green]RAG System setup complete![/bold green]")
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Chat with the RAG system"""
        if not self.rag_chain:
            raise ValueError("System not setup. Call setup_system() first.")
        
        return self.rag_chain.query(question)
    
    def evaluate(self, test_questions: List[str]):
        """Evaluate system performance"""
        if not self.evaluator:
            raise ValueError("System not setup. Call setup_system() first.")
        
        return self.evaluator.evaluate_responses(test_questions)

# Example usage
if __name__ == "__main__":
    # Initialize system
    config = RAGConfig(
        chunk_size=1000,
        top_k=5,
        temperature=0.1
    )
    
    rag_system = RAGSystem(config)
    
    # Setup (uncomment when ready)
    # rag_system.setup_system("./documents")
    
    # Example chat
    # result = rag_system.chat("What are the key findings in the research papers?")
    # console.print(result)
    
    print("Advanced RAG System initialized! Use setup_system() to get started.")