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
#from qdrant_client import QdrantClient
#from qdrant_client.models import Distance, VectorParams
from langchain_community.vectorstores import Chroma  # Add this line

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
        self.embedding_model = "text-embedding-3-large"  # CHANGE THIS
        self.llm_model = "gpt-4-turbo-preview"  # ALREADY CORRECT
        self.chunk_size = 1500  # CHANGE FROM 1000 TO 1500
        self.chunk_overlap = 300  # CHANGE FROM 200 TO 300
        self.top_k = 15  # CHANGE FROM 5 TO 15
        self.temperature = 0.1

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
        
        
        # Create vector store (in-memory for deployment)
        with st.spinner("Creating embeddings and storing in vector database..."):
            from langchain_community.vectorstores import Chroma
            vector_store = Chroma.from_documents(
                chunks,
                embeddings
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

        # Deduplicate by content instead of chunk_id
        seen_content = set()
        unique_docs = []
        for doc in docs:
            # Use first 100 characters as content fingerprint
            content_fingerprint = doc.page_content[:100]
            if content_fingerprint not in seen_content:
                seen_content.add(content_fingerprint)
                unique_docs.append(doc)
        
        docs = unique_docs
        
        if not docs:
            return {
                'answer': "I couldn't find relevant information in the documents to answer your question.",
                'sources': [],
                'processing_time': time.time() - start_time
            }
        
        # Rest of your code stays the same...
        
        # Create context
        context = "\n\n---\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Create prompt
        prompt = f"""You are an AI assistant with access to transcribed conversations between Gerro and Angus about rugby strategy and game planning. Angus is a rugby expert who shared his knowledge with Gerro to help create this system.
Your role is to act as Angus's rugby knowledge base, providing expert advice to people who may not know rugby well. Answer questions as if you're sharing Angus's insights and expertise about practice sessions, game strategy, and rugby fundamentals.
Based on the following conversation excerpts between Gerro and Angus, provide helpful rugby advice and guidance. Try to use the documents info as much as possible compared to just general rugby knowledge. We specificily want to pick Angus's brain.
IMPORTANT: Keep your responses concise and direct. Aim for short answers, with a few sentances. Focus on the most actionable advice. Don't mention that we're using this in a RAG system just focus on the topic of rugby advice.


Here is the document Excerpts:
{context}

Here is the RAG app Question: {question}

Please provide a detailed answer based on the information available in the documents.

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
    st.markdown('<h1 class="main-header">🧠 Rugby Coach AI - Powered by Angus\'s Expertise</h1>', unsafe_allow_html=True)
    st.markdown("*Optimized demo configuration with GPT-4 Turbo and maximum retrieval power*")
    
    # Check API key first
    if not check_api_key():
        st.error("⚠️ OpenAI API key not configured. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=sk-your-key-here")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # Document management only
        st.subheader("📁 Document Management")
        
        # Show current documents
        documents_path = "./documents"
        if os.path.exists(documents_path):
            doc_files = [f for f in os.listdir(documents_path) 
                        if os.path.isfile(os.path.join(documents_path, f)) 
                        and not f.startswith('.') 
                        and not f.startswith('__')]
            st.write(f"📄 Documents found: {len(doc_files)}")
            if doc_files:
                with st.expander("View documents"):
                    for doc in doc_files:
                        file_path = os.path.join(documents_path, doc)
                        try:
                            file_size = os.path.getsize(file_path)
                            st.text(f"• {doc} ({file_size} bytes)")
                        except:
                            st.text(f"• {doc}")
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary"):
            config = RAGConfig()  # Use default optimal settings
            
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
            
            # Show optimized system stats
            st.subheader("📊 System Configuration")
            st.markdown("""
            **Optimized for Demo:**
            - **Model:** GPT-4 Turbo (Most Powerful)
            - **Embeddings:** text-embedding-3-large
            - **Chunk Size:** 1500 characters
            - **Retrieval:** Top 15 most relevant chunks
            - **Documents:** """ + str(st.session_state.documents_processed) + """
            - **Chunks:** """ + str(st.session_state.chunks_created))
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
