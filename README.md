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
