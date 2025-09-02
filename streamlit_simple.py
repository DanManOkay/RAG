import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

load_dotenv()

st.title("Simple RAG Test")

# Initialize
if "system_ready" not in st.session_state:
    with st.spinner("Setting up..."):
        client = QdrantClient(url="http://localhost:6333")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
        
        vector_store = Qdrant(
            client=client,
            collection_name="documents", 
            embeddings=embeddings
        )
        
        st.session_state.vector_store = vector_store
        st.session_state.llm = llm
        st.session_state.system_ready = True

if st.session_state.system_ready:
    question = st.text_input("Ask a question:")
    
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            # Get relevant documents
            docs = st.session_state.vector_store.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
            
            # Get response
            response = st.session_state.llm.invoke(prompt)
            st.write("**Answer:**", response.content)
            
            # Show sources
            with st.expander("Source Documents"):
                for i, doc in enumerate(docs):
                    st.write(f"**Source {i+1}:**")
                    st.write(doc.page_content[:200] + "...")
