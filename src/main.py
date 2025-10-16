"""
Main entry point for RAG Agent
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings
from core.rag_engine import RAGEngine
from utils.logging_utils import setup_logging


def main():
    """Main function to run the RAG Agent"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        settings = Settings()
        logger.info("Configuration loaded successfully")
        
        # Initialize RAG engine
        rag_engine = RAGEngine(settings)
        logger.info("RAG Engine initialized successfully")
        
        # Example usage
        query = "What is the main topic of the documents?"
        result = rag_engine.query(query)
        
        print(f"Query: {query}")
        print(f"Answer: {result}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
