"""
Environment setup script for RAG Agent
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        "data",
        "data/input",
        "data/output",
        "data/vector_store",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )


def setup_environment_file():
    """Setup environment file from template"""
    print("⚙️ Setting up environment file...")
    
    env_example = Path("env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys and settings")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("❌ env.example not found")


def verify_installation():
    """Verify that key packages are installed"""
    print("🔍 Verifying installation...")
    
    packages_to_check = [
        "pydantic",
        "numpy",
        "sentence_transformers",
        "openai",
        "chromadb"
    ]
    
    all_installed = True
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            all_installed = False
    
    return all_installed


def main():
    """Main setup function"""
    print("🚀 Setting up RAG Agent environment...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup environment file
    setup_environment_file()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 RAG Agent setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Add documents to data/input/ directory")
    print("3. Run: python src/main.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
