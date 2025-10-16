# RAG Agent

A powerful Retrieval-Augmented Generation (RAG) system for document processing and intelligent question answering.

## 🚀 Features

- **Document Processing**: Support for multiple document formats (TXT, PDF, DOCX)
- **Intelligent Chunking**: Smart document segmentation with configurable chunk sizes
- **Vector Storage**: Multiple vector database backends (ChromaDB, FAISS)
- **Embedding Models**: Integration with sentence-transformers for text embeddings
- **LLM Integration**: OpenAI GPT models for response generation
- **Modular Architecture**: Clean, extensible codebase with clear separation of concerns
- **Comprehensive Testing**: Full test suite with pytest
- **Easy Deployment**: Docker support and deployment scripts

## 📁 Project Structure

```
my_rag_project/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── main.py                   # Entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Configuration management
│   ├── core/                     # Core system components
│   │   ├── __init__.py
│   │   ├── document_processor.py # Document processing logic
│   │   └── rag_engine.py         # Main RAG orchestrator
│   ├── modules/                  # Specialized modules
│   │   ├── __init__.py
│   │   ├── parser/               # Document parsers
│   │   ├── nlp/                  # NLP components
│   │   └── llm/                  # LLM integration
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── file_utils.py         # File operations
│   │   ├── logging_utils.py      # Logging configuration
│   │   └── vector_store.py       # Vector database operations
│   └── models/                   # Data models
│       ├── __init__.py
│       └── schemas.py            # Pydantic schemas
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_document_processor.py
│   └── test_rag_engine.py
├── data/                         # Data directory (git-ignored)
│   ├── input/                    # Input documents
│   └── output/                   # Generated outputs
├── notebooks/                    # Jupyter notebooks
│   └── exploration.ipynb         # Example usage
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
│   ├── setup_env.py              # Environment setup
│   └── deploy.py                 # Deployment script
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Modern Python configuration
├── env.example                  # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag_agent
   ```

2. **Run the setup script**:
   ```bash
   python scripts/setup_env.py
   ```

3. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys and settings
   ```

4. **Install dependencies manually** (if setup script fails):
   ```bash
   pip install -r requirements.txt
   ```

### Manual Installation

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env file with your configuration
   ```

## ⚙️ Configuration

Create a `.env` file based on `env.example`:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Model Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Store
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_PATH=data/vector_store

# Logging
LOG_LEVEL=INFO
LOG_FILE=data/rag_agent.log
```

## 🚀 Usage

### Basic Usage

1. **Add documents to the knowledge base**:
   ```bash
   # Place your documents in data/input/
   cp your_documents.pdf data/input/
   ```

2. **Run the RAG Agent**:
   ```bash
   python src/main.py
   ```

### Programmatic Usage

```python
from src.config.settings import Settings
from src.core.rag_engine import RAGEngine
from pathlib import Path

# Initialize
settings = Settings()
rag_engine = RAGEngine(settings)

# Add documents
documents = [Path("data/input/document1.pdf"), Path("data/input/document2.txt")]
rag_engine.add_documents(documents)

# Query the system
response = rag_engine.query("What is the main topic of the documents?")
print(response.answer)
```

### Jupyter Notebook

Open `notebooks/exploration.ipynb` for interactive examples and exploration.

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_rag_engine.py
```

## 🐳 Docker Deployment

### Using Docker Compose

1. **Build and run**:
   ```bash
   docker-compose up --build
   ```

2. **Run in background**:
   ```bash
   docker-compose up -d
   ```

### Manual Docker

1. **Build image**:
   ```bash
   docker build -t rag-agent .
   ```

2. **Run container**:
   ```bash
   docker run -d \
     --name rag-agent \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/logs:/app/logs \
     --env-file .env \
     rag-agent
   ```

## 📊 Supported Document Types

- **Text files** (`.txt`)
- **PDF documents** (`.pdf`)
- **Word documents** (`.docx`)
- **Markdown files** (`.md`)

## 🔧 Architecture

### Core Components

1. **Document Processor**: Handles document parsing and chunking
2. **Embedder**: Generates text embeddings using sentence-transformers
3. **Vector Store**: Manages vector database operations
4. **LLM Client**: Interfaces with OpenAI's GPT models
5. **RAG Engine**: Orchestrates the entire pipeline

### Data Flow

1. **Document Ingestion**: Documents are processed and chunked
2. **Embedding Generation**: Text chunks are converted to vectors
3. **Vector Storage**: Embeddings are stored in vector database
4. **Query Processing**: User queries are embedded and matched
5. **Response Generation**: LLM generates answers using retrieved context

## 🛠️ Development

### Code Quality Standards

The project enforces strict code quality standards:

- **Black** for code formatting (88 character line length)
- **isort** for import sorting
- **Flake8** for linting
- **MyPy** for type checking
- **Bandit** for security checks
- **Pytest** for testing (80%+ coverage required)

### Quick Setup

```bash
# Set up development environment
python scripts/setup_dev_environment.py

# Or use Make commands
make setup-dev
```

### Quality Checks

```bash
# Run all quality checks
make check-quality

# Auto-fix formatting issues
make fix-formatting

# Run specific checks
make lint
make test
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Development Workflow

1. **Create feature branch**
2. **Write code** following standards
3. **Add tests** for new functionality
4. **Run quality checks**: `make check-quality`
5. **Commit changes** (hooks will run automatically)
6. **Create Pull Request**

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed guidelines.

### Adding New Document Types

1. Create a new processor class in `src/core/document_processor.py`
2. Implement the `DocumentProcessor` interface
3. Add the processor to `DocumentProcessorFactory`
4. Add tests in `tests/test_document_processor.py`

## 📈 Performance Considerations

- **Chunk Size**: Larger chunks provide more context but may reduce precision
- **Embedding Model**: Choose based on your language and domain requirements
- **Vector Store**: ChromaDB for development, FAISS for production scale
- **LLM Model**: Balance between cost and quality based on your needs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **API Key Issues**: Verify your OpenAI API key is correct
3. **Memory Issues**: Reduce chunk size or use smaller embedding models
4. **Vector Store Errors**: Clear the vector store directory and restart

### Getting Help

- Check the logs in `data/rag_agent.log`
- Review the test cases for usage examples
- Open an issue on GitHub for bugs or feature requests

## 🔮 Roadmap

- [ ] Web interface for document upload and querying
- [ ] Support for more document formats (HTML, XML, etc.)
- [ ] Advanced chunking strategies (semantic, hierarchical)
- [ ] Integration with more LLM providers
- [ ] Real-time document processing
- [ ] Multi-language support
- [ ] Advanced analytics and monitoring

---

**Happy RAG-ing! 🎉**
