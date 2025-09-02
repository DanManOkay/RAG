#!/bin/bash

# Advanced RAG System Setup Script
echo "🚀 Setting up Advanced RAG System in GitHub Codespace..."

# Create project structure
echo "📁 Creating project structure..."
mkdir -p documents
mkdir -p qdrant_storage
mkdir -p .devcontainer

# Create environment file
echo "⚙️ Creating environment configuration..."
cat > .env << EOF
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Vector Database Configuration
QDRANT_URL=http://localhost:6333

# LangSmith (optional - for monitoring)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=advanced-rag

# Cohere (optional - for alternative embeddings)
COHERE_API_KEY=your_cohere_api_key_here

# Pinecone (optional - alternative vector DB)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment

# Weights & Biases (optional - for experiment tracking)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=rag-system
EOF

# Create gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << EOF
# Environment variables
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# Vector database storage
qdrant_storage/
chroma_db/

# Documents (you may want to remove this line to include your docs)
documents/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Model cache
.cache/
models/
EOF

# Create README
echo "📖 Creating README..."
cat > README.md << 'EOF'
# Advanced RAG System

A high-quality Retrieval-Augmented Generation (RAG) system built with the best frameworks available.

## Features

- **Multiple Vector Databases**: Qdrant, Chroma, Pinecone, Weaviate support
- **Advanced Document Processing**: Smart chunking with multiple strategies
- **Hybrid Search**: Combines semantic and keyword search
- **Conversation Memory**: Maintains context across multiple exchanges
- **RAG Evaluation**: Built-in evaluation metrics using RAGAS
- **Web Interface**: Beautiful Streamlit chat interface
- **Monitoring**: Integration with LangSmith and W&B
- **Multiple File Types**: PDF, DOCX, PPTX, TXT, CSV, and more

## Quick Start

1. **Setup Environment**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Configure API Keys**:
   Edit `.env` file with your API keys:
   ```bash
   OPENAI_API_KEY=your_key_here
   ```

3. **Start Vector Database**:
   ```bash
   docker-compose up -d
   ```

4. **Add Documents**:
   Place your documents in the `documents/` folder

5. **Run the System**:
   ```bash
   streamlit run streamlit_app.py
   ```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Processing    │───▶│  Vector Store   │
│   (PDF, DOCX)   │    │   (Chunking)    │    │   (Qdrant)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   User Query    │───▶│   Retrieval     │◀────────────┘
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Context +     │───▶│   LLM Response  │
                       │   Query         │    │   (GPT-4)       │
                       └─────────────────┘    └─────────────────┘
```

## Configuration Options

### Vector Databases
- **Qdrant** (Recommended): Fast, scalable, with filtering
- **Chroma**: Lightweight, good for development
- **Pinecone**: Managed service, enterprise features
- **Weaviate**: GraphQL, semantic search

### Embedding Models
- `text-embedding-3-large`: Best quality (3072 dimensions)
- `text-embedding-3-small`: Good balance (1536 dimensions)
- `text-embedding-ada-002`: Legacy but reliable

### LLM Models
- `gpt-4-turbo-preview`: Best reasoning and context
- `gpt-4`: Reliable, high quality
- `gpt-3.5-turbo`: Fast and cost-effective

## Advanced Usage

### Custom Configuration
```python
from advanced_rag import RAGSystem, RAGConfig

config = RAGConfig(
    chunk_size=1500,
    chunk_overlap=300,
    top_k=10,
    temperature=0.0
)

rag_system = RAGSystem(config)
```

### Evaluation
```python
test_questions = [
    "What are the main findings?",
    "What recommendations are made?"
]

results = rag_system.evaluate(test_questions)
print(results)
```

### API Usage
```python
# Setup system
rag_system.setup_system("./documents")

# Chat
response = rag_system.chat("Your question here")
print(response["answer"])
```

## File Structure

```
advanced-rag/
├── .devcontainer/
│   └── devcontainer.json      # Codespace configuration
├── documents/                 # Your documents go here
├── qdrant_storage/           # Vector database storage
├── advanced_rag.py           # Main RAG system
├── streamlit_app.py          # Web interface
├── docker-compose.yml        # Database services
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── README.md                 # This file
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your OpenAI API key is set in `.env`
2. **Vector DB Connection**: Make sure Qdrant is running with `docker-compose up -d`
3. **Memory Issues**: Reduce `chunk_size` for large documents
4. **Slow Performance**: Use smaller embedding models or reduce `top_k`

### Performance Optimization

1. **Batch Processing**: Process documents in batches for large collections
2. **Caching**: Enable embedding caching for repeated queries
3. **Hardware**: Use GPU for faster embedding generation
4. **Indexing**: Optimize vector database indices for your use case

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details
EOF

echo "🐳 Starting Docker services..."
docker-compose up -d

echo "🔧 Installing Python dependencies..."
pip install -r requirements.txt

echo "📋 Creating sample documents..."
mkdir -p documents/samples
cat > documents/samples/sample_doc.txt << 'EOF'
# Sample Document for RAG System

## Introduction
This is a sample document to test the RAG (Retrieval-Augmented Generation) system. The system should be able to answer questions about the content in this document.

## Key Features
1. Advanced document processing with intelligent chunking
2. Multiple vector database support (Qdrant, Chroma, Pinecone)
3. Hybrid search capabilities combining semantic and keyword search
4. Conversation memory for maintaining context
5. Built-in evaluation metrics using RAGAS framework

## Technical Architecture
The system uses a multi-stage pipeline:
- Document loading and preprocessing
- Smart chunking based on document structure
- Embedding generation using state-of-the-art models
- Vector storage with optimized retrieval
- Context-aware response generation

## Benefits
- Improved accuracy through advanced retrieval methods
- Scalable architecture supporting large document collections
- User-friendly interface with Streamlit
- Comprehensive monitoring and evaluation tools

## Conclusion
This RAG system represents the state-of-the-art in document-based question answering, combining the best available frameworks and techniques.
EOF

echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys"
echo "2. Add your documents to the documents/ folder"
echo "3. Run: streamlit run streamlit_app.py"
echo "4. Open the web interface and initialize the system"
echo ""
echo "For API usage, see the README.md file"