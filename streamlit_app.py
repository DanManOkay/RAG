import streamlit as st
import os
from pathlib import Path
import time
from typing import List, Dict
import pandas as pd

# Import our RAG system
from advanced_rag import RAGSystem, RAGConfig

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Chat System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

def initialize_rag_system(config_params: Dict) -> RAGSystem:
    """Initialize the RAG system with given parameters"""
    config = RAGConfig(**config_params)
    return RAGSystem(config)

def setup_system(rag_system: RAGSystem, documents_path: str):
    """Setup the RAG system with documents"""
    with st.spinner("Setting up RAG system... This may take a few minutes."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Processing documents...")
            progress_bar.progress(25)
            
            status_text.text("Creating embeddings...")
            progress_bar.progress(50)
            
            rag_system.setup_system(documents_path)
            
            status_text.text("Finalizing setup...")
            progress_bar.progress(100)
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ RAG system setup complete!")
            return True
            
        except Exception as e:
            st.error(f"❌ Setup failed: {str(e)}")
            return False

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
        
        # Display source documents
        if 'source_documents' in message and message['source_documents']:
            with st.expander("📚 Source Documents", expanded=False):
                for i, doc in enumerate(message['source_documents']):
                    st.markdown(f"""
                    <div class="source-doc">
                        <strong>Source {i+1}:</strong> {doc['source']}<br>
                        <strong>Content:</strong> {doc['content']}
                    </div>
                    """, unsafe_allow_html=True)

# Main app
def main():
    st.markdown('<h1 class="main-header">🧠 Advanced RAG Chat System</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System configuration
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
        
        # Document upload
        st.subheader("📁 Document Management")
        
        documents_path = st.text_input("Documents Directory", value="./documents")
        
        if st.button("🚀 Initialize System", type="primary"):
            config_params = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "temperature": temperature,
                "embedding_model": embedding_model,
                "llm_model": llm_model
            }
            
            st.session_state.rag_system = initialize_rag_system(config_params)
            
            if os.path.exists(documents_path):
                if setup_system(st.session_state.rag_system, documents_path):
                    st.session_state.system_ready = True
            else:
                st.error(f"Directory {documents_path} does not exist!")
        
        # System status
        if st.session_state.system_ready:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ System Not Ready")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
    
    # Main chat interface
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
                        'content': user_input
                    })
                    
                    # Get response from RAG system
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.rag_system.chat(user_input)
                            
                            # Add assistant response to history
                            st.session_state.chat_history.append({
                                'type': 'assistant',
                                'data': response
                            })
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    
                    # Clear input and rerun
                    st.rerun()
            
            with col_example:
                if st.button("💡 Example Questions"):
                    examples = [
                        "What are the main topics covered in the documents?",
                        "Can you summarize the key findings?",
                        "What are the recommendations mentioned?",
                        "Are there any specific dates or numbers mentioned?"
                    ]
                    st.info("Try asking:\n" + "\n".join(f"• {ex}" for ex in examples))
        
        else:
            st.info("👈 Please configure and initialize the system in the sidebar first.")
    
    with col2:
        st.header("📊 Analytics")
        
        if st.session_state.system_ready and st.session_state.chat_history:
            # Chat statistics
            total_messages = len(st.session_state.chat_history)
            user_messages = len([m for m in st.session_state.chat_history if m['type'] == 'user'])
            
            st.metric("Total Messages", total_messages)
            st.metric("User Questions", user_messages)
            
            # Recent topics
            if user_messages > 0:
                st.subheader("Recent Questions")
                recent_questions = [
                    m['content'][:50] + "..." if len(m['content']) > 50 else m['content']
                    for m in st.session_state.chat_history[-6:]
                    if m['type'] == 'user'
                ]
                for q in recent_questions[-3:]:
                    st.text(f"• {q}")
        
        # System info
        st.subheader("ℹ️ System Info")
        if st.session_state.system_ready:
            config = st.session_state.rag_system.config
            st.text(f"Model: {config.llm_model}")
            st.text(f"Chunk Size: {config.chunk_size}")
            st.text(f"Top-K: {config.top_k}")
            st.text(f"Temperature: {config.temperature}")

if __name__ == "__main__":
    main()