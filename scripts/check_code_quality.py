"""
Script to check code quality and enforce coding standards
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class CodeQualityChecker:
    """Code quality checker and enforcer"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"

    def run_command(self, command: List[str], description: str) -> Tuple[bool, str]:
        """Run a command and return success status and output"""
        print(f"🔄 {description}...")
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ {description} passed")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed")
            print(f"Error: {e.stderr}")
            return False, e.stderr

    def check_black_formatting(self) -> bool:
        """Check code formatting with Black"""
        success, output = self.run_command(
            ["black", "--check", "--diff", "src/", "tests/"],
            "Black formatting check"
        )
        if not success:
            print("💡 Run 'black src/ tests/' to fix formatting issues")
        return success

    def check_isort_imports(self) -> bool:
        """Check import sorting with isort"""
        success, output = self.run_command(
            ["isort", "--check-only", "--diff", "src/", "tests/"],
            "Import sorting check"
        )
        if not success:
            print("💡 Run 'isort src/ tests/' to fix import issues")
        return success

    def check_flake8_linting(self) -> bool:
        """Check code with flake8"""
        success, output = self.run_command(
            ["flake8", "src/", "tests/"],
            "Flake8 linting check"
        )
        return success

    def check_mypy_types(self) -> bool:
        """Check type hints with mypy"""
        success, output = self.run_command(
            ["mypy", "src/"],
            "MyPy type checking"
        )
        return success

    def check_bandit_security(self) -> bool:
        """Check for security issues with bandit"""
        success, output = self.run_command(
            ["bandit", "-r", "src/", "-f", "json"],
            "Bandit security check"
        )
        return success

    def run_tests(self) -> bool:
        """Run the test suite"""
        success, output = self.run_command(
            ["pytest", "tests/", "-v", "--cov=src", "--cov-report=term-missing"],
            "Running tests"
        )
        return success

    def check_all(self) -> bool:
        """Run all quality checks"""
        print("🚀 Running comprehensive code quality checks...")
        print("=" * 60)

        checks = [
            ("Black formatting", self.check_black_formatting),
            ("Import sorting", self.check_isort_imports),
            ("Flake8 linting", self.check_flake8_linting),
            ("MyPy type checking", self.check_mypy_types),
            ("Bandit security", self.check_bandit_security),
            ("Test suite", self.run_tests),
        ]

        results = []
        for name, check_func in checks:
            try:
                result = check_func()
                results.append((name, result))
            except Exception as e:
                print(f"❌ {name} failed with exception: {e}")
                results.append((name, False))

        print("=" * 60)
        print("📊 Quality Check Results:")

        all_passed = True
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n🎉 All quality checks passed!")
        else:
            print("\n⚠️  Some quality checks failed. Please fix the issues above.")

        return all_passed

    def fix_formatting(self) -> bool:
        """Auto-fix formatting issues"""
        print("🔧 Auto-fixing formatting issues...")

        # Fix Black formatting
        success1, _ = self.run_command(
            ["black", "src/", "tests/"],
            "Fixing Black formatting"
        )

        # Fix import sorting
        success2, _ = self.run_command(
            ["isort", "src/", "tests/"],
            "Fixing import sorting"
        )

        return success1 and success2


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Code quality checker")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix formatting issues"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run all quality checks"
    )

    args = parser.parse_args()

    checker = CodeQualityChecker()

    if args.fix:
        success = checker.fix_formatting()
        sys.exit(0 if success else 1)
    elif args.check:
        success = checker.check_all()
        sys.exit(0 if success else 1)
    else:
        # Default: run all checks
        success = checker.check_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
