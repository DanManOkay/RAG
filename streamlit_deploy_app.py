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
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG System",
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
    .upload-section {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px dashed #1f77b4;
        margin: 1rem 0;
    }
    .prompt-section {
        background-color: #fff9e6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #ffc107;
        margin: 1rem 0;
    }
    .config-section {
        background-color: #f0fff4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Default system prompt - BLANK
DEFAULT_SYSTEM_PROMPT = ""

# Configuration class
class RAGConfig:
    def __init__(self, **kwargs):
        self.vector_db_url = kwargs.get('vector_db_url', "http://localhost:6333")
        self.collection_name = kwargs.get('collection_name', "documents")
        self.embedding_model = kwargs.get('embedding_model', "text-embedding-3-large")
        self.llm_model = kwargs.get('llm_model', "gpt-4-turbo-preview")
        self.chunk_size = kwargs.get('chunk_size', 1500)
        self.chunk_overlap = kwargs.get('chunk_overlap', 300)
        self.top_k = kwargs.get('top_k', 15)
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
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = "./temp_uploads"
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if "rag_config" not in st.session_state:
    st.session_state.rag_config = {
        'llm_model': 'gpt-4-turbo-preview',
        'embedding_model': 'text-embedding-3-large',
        'temperature': 0.1,
        'chunk_size': 1500,
        'chunk_overlap': 300,
        'top_k': 15
    }

def check_api_key():
    """Check if OpenAI API key is configured"""
    api_key = os.getenv('OPENAI_API_KEY')
    return api_key and api_key != "your_openai_api_key_here" and len(api_key) > 20

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory"""
    # Create temp directory if it doesn't exist
    temp_dir = st.session_state.temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    saved_files = []
    for uploaded_file in uploaded_files:
        try:
            # Save file
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Error saving {uploaded_file.name}: {str(e)}")
    
    return saved_files

def load_and_process_documents(documents_path: str, config: RAGConfig):
    """Load and process documents WITHOUT unstructured"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Load documents
        status_text.text("Loading documents...")
        progress_bar.progress(20)
        
        if not os.path.exists(documents_path):
            st.error(f"Directory not found: {documents_path}")
            return None, None
        
        # Get all files
        doc_files = []
        for ext in ['*.txt', '*.pdf', '*.docx', '*.doc']:
            doc_files.extend(Path(documents_path).glob(f"**/{ext}"))
        
        if not doc_files:
            st.error(f"No documents found in {documents_path}")
            return None, None
        
        # Load each file with appropriate loader
        documents = []
        for file_path in doc_files:
            try:
                status_text.text(f"Loading {file_path.name}...")
                
                if file_path.suffix.lower() == '.txt':
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    documents.extend(loader.load())
                    
                elif file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    documents.extend(loader.load())
                    
                elif file_path.suffix.lower() in ['.docx', '.doc']:
                    loader = Docx2txtLoader(str(file_path))
                    documents.extend(loader.load())
                    
            except Exception as e:
                st.warning(f"Could not load {file_path.name}: {str(e)}")
                continue
        
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
        status_text.text(f"✅ Loaded {len(documents)} documents into {len(chunks)} chunks!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return documents, chunks
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return None, None

def setup_rag_system(config: RAGConfig, documents_path: str):
    """Setup the RAG system"""
    try:
        # Initialize components
        embeddings = OpenAIEmbeddings(model=config.embedding_model)
        llm = ChatOpenAI(model=config.llm_model, temperature=config.temperature)
        
        # Load and process documents
        documents, chunks = load_and_process_documents(documents_path, config)
        
        if not chunks:
            return None
        
        # Create vector store with FAISS
        with st.spinner("Creating embeddings and storing in vector database..."):
            vector_store = FAISS.from_documents(chunks, embeddings)
        
        return {
            'vector_store': vector_store,
            'llm': llm,
            'config': config,
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

        # Deduplicate by content
        seen_content = set()
        unique_docs = []
        for doc in docs:
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
        
        # Create context
        context = "\n\n---\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Use custom system prompt from session state
        prompt = f"""{st.session_state.system_prompt}

Here are the document Excerpts:
{context}

Here is the Question: {question}

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
    st.markdown('<h1 class="main-header">🧠 RAG Tester</h1>', unsafe_allow_html=True)
    st.markdown("*Customizable document Q&A*")
    
    # Check API key first
    if not check_api_key():
        st.error("⚠️ OpenAI API key not configured. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=sk-your-key-here")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # Document Upload Section
        st.subheader("📤 Step 1: Upload Documents")
        
        st.markdown("""
        <div class="upload-section">
        Upload your documents here.<br>
        Supported formats: PDF, DOCX, DOC, TXT
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Show uploaded files
            with st.expander("View uploaded files"):
                for file in uploaded_files:
                    file_size = len(file.getvalue())
                    st.text(f"• {file.name} ({file_size:,} bytes)")
            
            # Save files button
            if st.button("💾 Save Uploaded Files", type="secondary"):
                with st.spinner("Saving files..."):
                    saved_files = save_uploaded_files(uploaded_files)
                    st.session_state.uploaded_files = saved_files
                    st.success(f"✅ Saved {len(saved_files)} file(s)")
                    st.rerun()
        
        # Show currently saved files
        if st.session_state.uploaded_files:
            st.write(f"📄 Documents ready: {len(st.session_state.uploaded_files)}")
            with st.expander("View saved documents"):
                for doc in st.session_state.uploaded_files:
                    st.text(f"• {doc}")
        
        st.divider()
        
        # System Prompt Configuration
        st.subheader("✏️ Step 2: Configure System Prompt")
        
        st.markdown("""
        <div class="prompt-section">
        Customize how the AI assistant should behave and respond.
        </div>
        """, unsafe_allow_html=True)
        
        # Text area for system prompt
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=250,
            placeholder="Write your custom system prompt here...\n\nExample:\nYou are an AI assistant with access to [document type].\nProvide [type of responses] based on the documents.\nKeep responses [tone/style].",
            help="This prompt defines how the AI assistant will behave. It will be prepended to every query.",
            key="prompt_input"
        )
        
        # Update prompt button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Prompt", type="secondary"):
                st.session_state.system_prompt = system_prompt
                st.success("✅ Prompt saved!")
        
        with col2:
            if st.button("🔄 Reset to Default"):
                st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
                st.success("✅ Reset to default!")
                st.rerun()
        
        # Show current prompt info
        prompt_length = len(st.session_state.system_prompt)
        st.caption(f"Current prompt: {prompt_length} characters")
        
        st.divider()
        
        # Advanced Configuration Section
        st.subheader("⚙️ Step 3: RAG Settings")
        
        st.markdown("""
        <div class="config-section">
        Adjust RAG system parameters for optimal performance.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔧 Configuration", expanded=False):
            # Model Settings
            st.markdown("**🤖 Model Configuration**")
            
            llm_model = st.selectbox(
                "LLM Model",
                options=[
                    "gpt-4-turbo-preview",
                    "gpt-4o",
                    "gpt-5-nano"
                ],
                index=0,
                help="The AI model that generates responses. GPT-4 Turbo is most powerful."
            )
            
            embedding_model = st.selectbox(
                "Embedding Model",
                options=[
                    "text-embedding-3-large",
                    "text-embedding-3-small",
                    "text-embedding-ada-002"
                ],
                index=0,
                help="Model that converts text to vectors. 3-large is most accurate but slower."
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.rag_config.get('temperature', 0.1),
                step=0.1,
                help="Controls randomness. 0.0 = deterministic, 2.0 = very creative."
            )
            
            st.divider()
            
            # Chunking Settings
            st.markdown("**📏 Document Processing**")
            
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=500,
                max_value=3000,
                value=st.session_state.rag_config.get('chunk_size', 1500),
                step=100,
                help="Size of text chunks. Larger = more context, smaller = more precise."
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=500,
                value=st.session_state.rag_config.get('chunk_overlap', 300),
                step=50,
                help="How much chunks overlap. Prevents splitting important info."
            )
            
            st.divider()
            
            # Retrieval Settings
            st.markdown("**🔍 Retrieval Configuration**")
            
            top_k = st.slider(
                "Top-K Retrieval",
                min_value=1,
                max_value=30,
                value=st.session_state.rag_config.get('top_k', 15),
                step=1,
                help="Number of relevant chunks to retrieve. More = comprehensive but slower."
            )
            
            st.divider()
            
            # Save configuration button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Settings", key="save_rag_config"):
                    st.session_state.rag_config = {
                        'llm_model': llm_model,
                        'embedding_model': embedding_model,
                        'temperature': temperature,
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap,
                        'top_k': top_k
                    }
                    st.success("✅ Settings saved!")
            
            with col2:
                if st.button("🔄 Reset to Defaults", key="reset_rag_config"):
                    st.session_state.rag_config = {
                        'llm_model': 'gpt-4-turbo-preview',
                        'embedding_model': 'text-embedding-3-large',
                        'temperature': 0.1,
                        'chunk_size': 1500,
                        'chunk_overlap': 300,
                        'top_k': 15
                    }
                    st.success("✅ Reset to defaults!")
                    st.rerun()
        
        # Show current configuration
        st.caption("**Current Settings:**")
        config = st.session_state.rag_config
        st.caption(f"• Model: {config['llm_model']}")
        st.caption(f"• Embeddings: {config['embedding_model']}")
        st.caption(f"• Temperature: {config['temperature']}")
        st.caption(f"• Chunk Size: {config['chunk_size']}")
        st.caption(f"• Overlap: {config['chunk_overlap']}")
        st.caption(f"• Top-K: {config['top_k']}")
        
        st.divider()
        
        # Initialize system button
        st.subheader("🚀 Step 4: Initialize System")
        
        if st.session_state.uploaded_files and st.session_state.system_prompt:
            if st.button("🚀 Initialize RAG System", type="primary"):
                # Use custom config from session state
                config = RAGConfig(**st.session_state.rag_config)
                
                with st.spinner("Initializing RAG system..."):
                    rag_system = setup_rag_system(config, st.session_state.temp_dir)
                    
                    if rag_system:
                        st.session_state.rag_system = rag_system
                        st.session_state.system_ready = True
                        st.success("✅ System initialized successfully!")
                        st.rerun()
        else:
            if not st.session_state.uploaded_files:
                st.info("👆 Please upload and save documents first")
            if not st.session_state.system_prompt:
                st.info("👆 Please configure system prompt")
        
        # System status
        if st.session_state.system_ready:
            st.success("✅ System Ready")
            
            # Show system stats
            st.subheader("📊 System Status")
            config = st.session_state.rag_system['config']
            st.markdown(f"""
            **Current Configuration:**
            - **Model:** {config.llm_model}
            - **Embeddings:** {config.embedding_model}
            - **Temperature:** {config.temperature}
            - **Chunk Size:** {config.chunk_size} chars
            - **Chunk Overlap:** {config.chunk_overlap} chars
            - **Retrieval:** Top {config.top_k} chunks
            - **Documents:** {st.session_state.documents_processed}
            - **Chunks:** {st.session_state.chunks_created}
            - **Prompt:** {len(st.session_state.system_prompt)} chars
            """)
        else:
            st.warning("⚠️ System Not Ready")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Reset system button
        if st.button("🔄 Reset Everything", type="secondary"):
            st.session_state.rag_system = None
            st.session_state.chat_history = []
            st.session_state.system_ready = False
            st.session_state.documents_processed = 0
            st.session_state.chunks_created = 0
            st.session_state.uploaded_files = []
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
            st.session_state.rag_config = {
                'llm_model': 'gpt-4-turbo-preview',
                'embedding_model': 'text-embedding-3-large',
                'temperature': 0.1,
                'chunk_size': 1500,
                'chunk_overlap': 300,
                'top_k': 15
            }
            if os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir)
            st.success("✅ System reset!")
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
            st.info("👈 Please complete all setup steps in the sidebar first.")
    
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
            st.text("Database: FAISS (In-Memory)")
        else:
            st.text("System not initialized")

if __name__ == "__main__":
    main()