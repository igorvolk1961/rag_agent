"""
Script to set up development environment with all quality tools
"""

import subprocess
import sys
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


def setup_development_environment():
    """Set up complete development environment"""
    print("🚀 Setting up RAG Agent development environment...")
    print("=" * 60)

    # Check Python version
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")

    # Install development dependencies
    if not run_command(
        f"{sys.executable} -m pip install -e \".[dev]\"",
        "Installing development dependencies"
    ):
        return False

    # Install pre-commit
    if not run_command(
        f"{sys.executable} -m pip install pre-commit",
        "Installing pre-commit"
    ):
        return False

    # Install pre-commit hooks
    if not run_command(
        "pre-commit install",
        "Installing pre-commit hooks"
    ):
        return False

    # Create necessary directories
    directories = [
        "data/input",
        "data/output",
        "data/vector_store",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Run initial code quality check
    print("\n🔍 Running initial code quality check...")
    if not run_command(
        f"{sys.executable} scripts/check_code_quality.py --check",
        "Initial quality check"
    ):
        print("⚠️  Some quality checks failed. This is normal for initial setup.")
        print("💡 Run 'make fix-formatting' to auto-fix formatting issues.")

    print("=" * 60)
    print("🎉 Development environment setup completed!")
    print("\nNext steps:")
    print("1. Configure your IDE (VS Code settings are in .vscode/)")
    print("2. Copy env.example to .env and configure your API keys")
    print("3. Run 'make check-quality' to verify everything works")
    print("4. Start developing!")
    print("\nUseful commands:")
    print("  make check-quality    - Run all quality checks")
    print("  make fix-formatting   - Auto-fix formatting issues")
    print("  make test            - Run tests")
    print("  make dev             - Run full development workflow")

    return True


def main():
    """Main function"""
    success = setup_development_environment()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
