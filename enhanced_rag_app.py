import streamlit as st
import os
import shutil
from pathlib import Path
import time
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Core imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load environment
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enhanced RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-doc {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Configuration class
class RAGConfig:
    def __init__(self, **kwargs):
        self.vector_db_url = kwargs.get('vector_db_url', "http://localhost:6333")
        self.collection_name = kwargs.get('collection_name', "documents")
        self.embedding_model = kwargs.get('embedding_model', "text-embedding-3-large")
        self.llm_model = kwargs.get('llm_model', "gpt-4-turbo-preview")
        self.chunk_size = kwargs.get('chunk_size', 1000)
        self.chunk_overlap = kwargs.get('chunk_overlap', 200)
        self.top_k = kwargs.get('top_k', 5)
        self.temperature = kwargs.get('temperature', 0.1)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = 0
if "chunks_created" not in st.session_state:
    st.session_state.chunks_created = 0

def check_api_key():
    """Check if OpenAI API key is configured"""
    api_key = os.getenv('OPENAI_API_KEY')
    return api_key and api_key != "your_openai_api_key_here" and len(api_key) > 20

def initialize_qdrant_collection(client, collection_name, embedding_dim=3072):
    """Initialize Qdrant collection"""
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            return f"Created new collection: {collection_name}"
        else:
            return f"Using existing collection: {collection_name}"
    except Exception as e:
        return f"Error with collection: {str(e)}"

def load_and_process_documents(documents_path: str, config: RAGConfig):
    """Load and process documents"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Load documents
        status_text.text("Loading documents...")
        progress_bar.progress(20)
        
        if not os.path.exists(documents_path) or not os.listdir(documents_path):
            st.error(f"No documents found in {documents_path}")
            return None, None
        
        loader = DirectoryLoader(
            documents_path,
            glob="**/*",
            loader_cls=UnstructuredFileLoader,
            show_progress=True
        )
        documents = loader.load()
        
        if not documents:
            st.error("No documents could be loaded")
            return None, None
        
        # Process documents
        status_text.text("Processing and chunking documents...")
        progress_bar.progress(40)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content),
                'processed_at': datetime.now().isoformat()
            })
        
        progress_bar.progress(60)
        
        st.session_state.documents_processed = len(documents)
        st.session_state.chunks_created = len(chunks)
        
        progress_bar.progress(100)
        status_text.text("Documents processed successfully!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return documents, chunks
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return None, None

def setup_rag_system(config: RAGConfig):
    """Setup the RAG system"""
    try:
        # Initialize components
        client = QdrantClient(url=config.vector_db_url)
        embeddings = OpenAIEmbeddings(model=config.embedding_model)
        llm = ChatOpenAI(model=config.llm_model, temperature=config.temperature)
        
        # Initialize collection
        collection_status = initialize_qdrant_collection(client, config.collection_name)
        st.info(collection_status)
        
        # Load and process documents
        documents, chunks = load_and_process_documents("./documents", config)
        
        if not chunks:
            return None
        
        # Create vector store
        with st.spinner("Creating embeddings and storing in vector database..."):
            vector_store = Qdrant.from_documents(
                chunks,
                embeddings,
                url=config.vector_db_url,
                collection_name=config.collection_name
            )
        
        return {
            'vector_store': vector_store,
            'llm': llm,
            'config': config,
            'client': client,
            'embeddings': embeddings
        }
        
    except Exception as e:
        st.error(f"Setup failed: {str(e)}")
        return None

def query_rag_system(question: str, rag_system: Dict, show_sources: bool = True):
    """Query the RAG system and return response with sources"""
    try:
        start_time = time.time()
        
        # Get relevant documents
        docs = rag_system['vector_store'].similarity_search(
            question, 
            k=rag_system['config'].top_k
        )
        
        if not docs:
            return {
                'answer': "I couldn't find relevant information in the documents to answer your question.",
                'sources': [],
                'processing_time': time.time() - start_time
            }
        
        # Create context
        context = "\n\n---\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Create prompt
        prompt = f"""Based on the following document excerpts, provide a comprehensive and accurate answer to the question. If the information is not available in the documents, clearly state this.

Document Excerpts:
{context}

Question: {question}

Please provide a detailed answer based on the information available in the documents. If you reference specific information, mention which document number it comes from.

Answer:"""
        
        # Get response from LLM
        response = rag_system['llm'].invoke(prompt)
        
        # Format sources
        sources = []
        for i, doc in enumerate(docs):
            sources.append({
                'document_number': i + 1,
                'content': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 'Unknown')
            })
        
        return {
            'answer': response.content,
            'sources': sources,
            'processing_time': time.time() - start_time
        }
        
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return {
            'answer': f"Error processing question: {str(e)}",
            'sources': [],
            'processing_time': 0
        }

def display_chat_message(message: Dict, is_user: bool = True):
    """Display a chat message"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong> {message['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        # Display processing time
        if 'processing_time' in message:
            st.caption(f"⏱️ Response time: {message['processing_time']:.2f} seconds")
        
        # Display sources
        if message.get('sources') and len(message['sources']) > 0:
            with st.expander("📚 Source Documents", expanded=False):
                for source in message['sources']:
                    st.markdown(f"""
                    <div class="source-doc">
                        <strong>Document {source['document_number']}:</strong> {source['source']}<br>
                        <strong>Content:</strong> {source['content']}<br>
                        <small>Chunk ID: {source['chunk_id']}</small>
                    </div>
                    """, unsafe_allow_html=True)

# Main application
def main():
    st.markdown('<h1 class="main-header">🧠 Enhanced RAG System</h1>', unsafe_allow_html=True)
    
    # Check API key first
    if not check_api_key():
        st.error("⚠️ OpenAI API key not configured. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=sk-your-key-here")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System settings
        st.subheader("System Settings")
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        top_k = st.slider("Retrieval K", min_value=1, max_value=20, value=5)
        temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        
        # Model selection
        st.subheader("Models")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]
        )
        
        llm_model = st.selectbox(
            "LLM Model", 
            ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"]
        )
        
        # Document management
        st.subheader("📁 Document Management")
        
        # Show current documents
        documents_path = "./documents"
        if os.path.exists(documents_path):
            doc_files = [f for f in os.listdir(documents_path) if os.path.isfile(os.path.join(documents_path, f))]
            st.write(f"📄 Documents found: {len(doc_files)}")
            if doc_files:
                with st.expander("View documents"):
                    for doc in doc_files:
                        st.text(f"• {doc}")
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary"):
            config = RAGConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                temperature=temperature,
                embedding_model=embedding_model,
                llm_model=llm_model
            )
            
            with st.spinner("Initializing RAG system..."):
                rag_system = setup_rag_system(config)
                
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.system_ready = True
                    st.success("✅ System initialized successfully!")
                    st.rerun()
        
        # System status
        if st.session_state.system_ready:
            st.success("✅ System Ready")
            
            # Show system stats
            st.subheader("📊 System Stats")
            st.markdown(f"""
            <div class="metric-card">
                <strong>Documents:</strong> {st.session_state.documents_processed}<br>
                <strong>Chunks:</strong> {st.session_state.chunks_created}<br>
                <strong>Model:</strong> {llm_model}<br>
                <strong>Embedding:</strong> {embedding_model}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ System Not Ready")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                display_chat_message({'content': message['content']}, is_user=True)
            else:
                display_chat_message(message['data'], is_user=False)
        
        # Chat input
        if st.session_state.system_ready:
            user_input = st.text_input(
                "Ask a question about your documents:",
                placeholder="What would you like to know?",
                key="user_input"
            )
            
            col_send, col_example = st.columns([1, 2])
            
            with col_send:
                if st.button("Send", type="primary") and user_input:
                    # Add user message to history
                    st.session_state.chat_history.append({
                        'type': 'user',
                        'content': user_input,
                        'timestamp': datetime.now()
                    })
                    
                    # Get response from RAG system
                    with st.spinner("Thinking..."):
                        response = query_rag_system(user_input, st.session_state.rag_system)
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({
                            'type': 'assistant',
                            'data': response,
                            'timestamp': datetime.now()
                        })
                    
                    st.rerun()
            
            with col_example:
                if st.button("💡 Example Questions"):
                    examples = [
                        "What are the main topics covered in the documents?",
                        "Can you provide a summary of the key findings?",
                        "What recommendations are mentioned?",
                        "Are there any specific dates or numbers mentioned?"
                    ]
                    st.info("Try asking:\n" + "\n".join(f"• {ex}" for ex in examples))
        
        else:
            st.info("👈 Please configure and initialize the system in the sidebar first.")
    
    with col2:
        st.header("📈 Analytics")
        
        if st.session_state.system_ready and st.session_state.chat_history:
            # Chat statistics
            total_messages = len(st.session_state.chat_history)
            user_messages = len([m for m in st.session_state.chat_history if m['type'] == 'user'])
            
            st.metric("Total Messages", total_messages)
            st.metric("Questions Asked", user_messages)
            
            # Response times
            response_times = [
                m['data']['processing_time'] 
                for m in st.session_state.chat_history 
                if m['type'] == 'assistant' and 'processing_time' in m['data']
            ]
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            
            # Recent questions
            st.subheader("Recent Questions")
            recent_questions = [
                m['content'][:50] + "..." if len(m['content']) > 50 else m['content']
                for m in st.session_state.chat_history[-6:]
                if m['type'] == 'user'
            ]
            for q in recent_questions[-3:]:
                st.text(f"• {q}")
        
        # System information
        st.subheader("ℹ️ System Info")
        if st.session_state.system_ready:
            config = st.session_state.rag_system['config']
            st.text(f"Chunk Size: {config.chunk_size}")
            st.text(f"Top-K: {config.top_k}")
            st.text(f"Temperature: {config.temperature}")
            
            # Vector database status
            try:
                client = st.session_state.rag_system['client']
                collections = client.get_collections()
                st.text(f"Collections: {len(collections.collections)}")
            except:
                st.text("Collections: Error")
        else:
            st.text("System not initialized")

if __name__ == "__main__":
    main()
