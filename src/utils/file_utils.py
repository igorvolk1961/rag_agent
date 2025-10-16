"""
File utility functions
"""

import logging
from pathlib import Path
from typing import List, Optional
import mimetypes


class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def read_text_file(file_path: Path) -> str:
        """Read a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    @staticmethod
    def read_pdf_file(file_path: Path) -> str:
        """Read a PDF file using PyPDF2 or similar"""
        try:
            import PyPDF2
            
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            return text.strip()
            
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        except Exception as e:
            raise Exception(f"Error reading PDF file: {e}")
    
    @staticmethod
    def get_supported_file_types() -> List[str]:
        """Get list of supported file types"""
        return ['.txt', '.pdf', '.md', '.docx']
    
    @staticmethod
    def is_supported_file(file_path: Path) -> bool:
        """Check if file type is supported"""
        return file_path.suffix.lower() in FileUtils.get_supported_file_types()
    
    @staticmethod
    def get_file_type(file_path: Path) -> str:
        """Get file type based on extension"""
        return file_path.suffix.lower()
    
    @staticmethod
    def get_mime_type(file_path: Path) -> Optional[str]:
        """Get MIME type of file"""
        return mimetypes.guess_type(str(file_path))[0]
    
    @staticmethod
    def find_files(directory: Path, extensions: List[str] = None) -> List[Path]:
        """Find files with specified extensions in directory"""
        if extensions is None:
            extensions = FileUtils.get_supported_file_types()
        
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"**/*{ext}"))
        
        return files
    
    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """Get file size in bytes"""
        return file_path.stat().st_size
    
    @staticmethod
    def get_file_info(file_path: Path) -> dict:
        """Get comprehensive file information"""
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "path": str(file_path),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "type": FileUtils.get_file_type(file_path),
            "mime_type": FileUtils.get_mime_type(file_path),
            "is_supported": FileUtils.is_supported_file(file_path)
        }
