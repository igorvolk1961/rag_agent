"""
Deployment script for RAG Agent
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any


class DeploymentManager:
    """Manages deployment of RAG Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = Path.cwd()
        self.deploy_dir = Path(config.get("deploy_dir", "deploy"))
    
    def clean_deploy_directory(self):
        """Clean deployment directory"""
        print("🧹 Cleaning deployment directory...")
        if self.deploy_dir.exists():
            shutil.rmtree(self.deploy_dir)
        self.deploy_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Cleaned {self.deploy_dir}")
    
    def copy_source_files(self):
        """Copy source files to deployment directory"""
        print("📁 Copying source files...")
        
        # Files and directories to copy
        items_to_copy = [
            "src/",
            "requirements.txt",
            "pyproject.toml",
            "README.md",
            "env.example"
        ]
        
        for item in items_to_copy:
            src = self.project_root / item
            dst = self.deploy_dir / item
            
            if src.is_file():
                shutil.copy2(src, dst)
                print(f"✅ Copied file: {item}")
            elif src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"✅ Copied directory: {item}")
            else:
                print(f"⚠️  Skipped (not found): {item}")
    
    def create_deployment_scripts(self):
        """Create deployment-specific scripts"""
        print("📝 Creating deployment scripts...")
        
        # Create startup script
        startup_script = self.deploy_dir / "start.sh"
        startup_content = """#!/bin/bash
# RAG Agent Startup Script

echo "🚀 Starting RAG Agent..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Activated virtual environment"
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/input data/output data/vector_store logs

# Start the application
echo "🎯 Starting RAG Agent..."
python src/main.py
"""
        startup_script.write_text(startup_content)
        startup_script.chmod(0o755)
        print("✅ Created start.sh")
        
        # Create Windows batch file
        startup_bat = self.deploy_dir / "start.bat"
        startup_bat_content = """@echo off
REM RAG Agent Startup Script for Windows

echo 🚀 Starting RAG Agent...

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
    echo ✅ Activated virtual environment
)

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
if not exist "data\\input" mkdir data\\input
if not exist "data\\output" mkdir data\\output
if not exist "data\\vector_store" mkdir data\\vector_store
if not exist "logs" mkdir logs

REM Start the application
echo 🎯 Starting RAG Agent...
python src\\main.py

pause
"""
        startup_bat.write_text(startup_bat_content)
        print("✅ Created start.bat")
    
    def create_dockerfile(self):
        """Create Dockerfile for containerized deployment"""
        print("🐳 Creating Dockerfile...")
        
        dockerfile_content = """FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .
COPY env.example .

# Create necessary directories
RUN mkdir -p data/input data/output data/vector_store logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (if needed for web interface)
EXPOSE 8000

# Default command
CMD ["python", "src/main.py"]
"""
        
        dockerfile_path = self.deploy_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        print("✅ Created Dockerfile")
    
    def create_docker_compose(self):
        """Create docker-compose.yml for easy deployment"""
        print("🐳 Creating docker-compose.yml...")
        
        compose_content = """version: '3.8'

services:
  rag-agent:
    build: .
    container_name: rag-agent
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}
    env_file:
      - .env
    restart: unless-stopped
    ports:
      - "8000:8000"
"""
        
        compose_path = self.deploy_dir / "docker-compose.yml"
        compose_path.write_text(compose_content)
        print("✅ Created docker-compose.yml")
    
    def create_readme(self):
        """Create deployment README"""
        print("📖 Creating deployment README...")
        
        readme_content = """# RAG Agent Deployment

This directory contains the deployment files for RAG Agent.

## Quick Start

### Option 1: Direct Python Execution

1. Copy your `.env` file to this directory
2. Run the startup script:
   - Linux/Mac: `./start.sh`
   - Windows: `start.bat`

### Option 2: Docker Deployment

1. Copy your `.env` file to this directory
2. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

### Option 3: Manual Docker

1. Build the image:
   ```bash
   docker build -t rag-agent .
   ```

2. Run the container:
   ```bash
   docker run -d \\
     --name rag-agent \\
     -v $(pwd)/data:/app/data \\
     -v $(pwd)/logs:/app/logs \\
     --env-file .env \\
     rag-agent
   ```

## Configuration

Make sure to configure the following in your `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `HUGGINGFACE_API_KEY`: Your Hugging Face API key (optional)
- `EMBEDDING_MODEL`: Embedding model to use
- `LLM_MODEL`: LLM model to use

## Data Directory Structure

```
data/
├── input/          # Place your documents here
├── output/         # Generated outputs
└── vector_store/   # Vector database files
```

## Logs

Application logs are stored in the `logs/` directory.

## Troubleshooting

1. Check the logs in `logs/` directory
2. Verify your API keys in `.env` file
3. Ensure you have sufficient disk space for vector store
4. Check that all required dependencies are installed
"""
        
        readme_path = self.deploy_dir / "README.md"
        readme_path.write_text(readme_content)
        print("✅ Created deployment README")
    
    def deploy(self):
        """Perform full deployment"""
        print("🚀 Starting deployment...")
        print("=" * 50)
        
        self.clean_deploy_directory()
        self.copy_source_files()
        self.create_deployment_scripts()
        self.create_dockerfile()
        self.create_docker_compose()
        self.create_readme()
        
        print("=" * 50)
        print("🎉 Deployment completed successfully!")
        print(f"📁 Deployment files created in: {self.deploy_dir}")
        print("\nNext steps:")
        print("1. Copy your .env file to the deployment directory")
        print("2. Add your documents to data/input/")
        print("3. Run the deployment using one of the methods in the README")


def main():
    """Main deployment function"""
    config = {
        "deploy_dir": "deploy",
        "include_tests": False,
        "include_docs": False
    }
    
    deployer = DeploymentManager(config)
    deployer.deploy()


if __name__ == "__main__":
    main()
