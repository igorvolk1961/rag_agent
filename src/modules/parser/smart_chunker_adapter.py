"""
Адаптер для интеграции SmartChanker с RAG Agent
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Отключаем аналитику unstructured
os.environ["UNSTRUCTURED_DISABLE_ANALYTICS"] = "true"

# Добавляем путь к smart_chanker в sys.path
smart_chanker_path = Path(__file__).parent.parent.parent.parent.parent / "smart_chanker"
if str(smart_chanker_path) not in sys.path:
    sys.path.insert(0, str(smart_chanker_path))

try:
    from smart_chanker import SmartChanker
    SMART_CHANKER_AVAILABLE = True
except ImportError as e:
    SMART_CHANKER_AVAILABLE = False
    logging.warning(f"SmartChanker не доступен: {e}")

import sys
from pathlib import Path

# Добавляем путь к src в sys.path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from models.schemas import Document, Chunk


class SmartChunkerAdapter:
    """
    Адаптер для интеграции SmartChanker с RAG Agent
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация адаптера

        Args:
            config_path: Путь к конфигурационному файлу SmartChanker
        """
        self.logger = logging.getLogger(__name__)

        if not SMART_CHANKER_AVAILABLE:
            raise ImportError("SmartChanker не установлен. Установите зависимости из requirements.txt")

        # Инициализируем SmartChanker
        self.smart_chanker = SmartChanker(config_path)

        # Настройки по умолчанию
        self.default_config = {
            "hierarchical_chunking": {
                "enabled": True,
                "target_level": 3,
                "max_chunk_size": 1000
            },
            "output": {
                "save_docx2python_text": False,
                "docx2python_text_suffix": "_docx2python.txt"
            }
        }

    def process_document(self, file_path: str) -> Document:
        """
        Обрабатывает документ с помощью SmartChanker

        Args:
            file_path: Путь к документу

        Returns:
            Document объект с чанками
        """
        try:
            self.logger.info(f"Обрабатываем документ с SmartChanker: {file_path}")

            # Запускаем полную обработку документа
            result = self.smart_chanker.run_end_to_end(file_path)

            # Извлекаем метаданные
            metadata = result.get("metadata", {})
            sections = result.get("sections", [])
            chunks_data = result.get("chunks", [])

            # Создаем чанки
            chunks = []
            for i, chunk_data in enumerate(chunks_data):
                chunk = Chunk(
                    content=chunk_data.get("content", ""),
                    metadata={
                        "chunk_id": chunk_data.get("id", f"chunk_{i}"),
                        "section_number": chunk_data.get("section_number", ""),
                        "level": chunk_data.get("level", 0),
                        "parent_sections": chunk_data.get("parent_sections", []),
                        "chunk_type": chunk_data.get("chunk_type", "content"),
                        "source_file": file_path,
                        **chunk_data.get("metadata", {})
                    }
                )
                chunks.append(chunk)

            # Создаем Document объект
            document = Document(
                file_path=file_path,
                content="\n\n".join([chunk.content for chunk in chunks]),
                chunks=chunks,
                metadata={
                    "source_file": file_path,
                    "total_chunks": len(chunks),
                    "sections_count": len(sections),
                    "processing_method": "smart_chunker",
                    "hierarchical_structure": True,
                    **metadata
                }
            )

            self.logger.info(f"Обработано {len(chunks)} чанков из {len(sections)} разделов")
            return document

        except Exception as e:
            self.logger.error(f"Ошибка обработки документа {file_path}: {e}")
            raise

    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Обрабатывает несколько документов

        Args:
            file_paths: Список путей к документам

        Returns:
            Список Document объектов
        """
        documents = []

        for file_path in file_paths:
            try:
                document = self.process_document(file_path)
                documents.append(document)
            except Exception as e:
                self.logger.error(f"Ошибка обработки документа {file_path}: {e}")
                # Создаем пустой документ с ошибкой
                error_document = Document(
                    file_path=file_path,
                    content="",
                    chunks=[],
                    metadata={
                        "source_file": file_path,
                        "error": str(e),
                        "processing_method": "smart_chunker"
                    }
                )
                documents.append(error_document)

        return documents

    def get_supported_extensions(self) -> List[str]:
        """
        Возвращает поддерживаемые расширения файлов

        Returns:
            Список поддерживаемых расширений
        """
        return ['.docx', '.doc', '.pdf', '.txt']

    def is_available(self) -> bool:
        """
        Проверяет доступность SmartChanker

        Returns:
            True если SmartChanker доступен
        """
        return SMART_CHANKER_AVAILABLE
