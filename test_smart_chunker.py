#!/usr/bin/env python3
"""
Тестовый скрипт для проверки интеграции SmartChunker
"""

import sys
from pathlib import Path

# Добавляем путь к src
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from modules.parser.smart_chunker_adapter import SmartChunkerAdapter

def test_smart_chunker():
    """Тестируем SmartChunker"""
    print("🧪 Тестирование SmartChunker...")

    try:
        print("Импортируем SmartChunkerAdapter...")
        # Инициализируем адаптер
        adapter = SmartChunkerAdapter("smart_chunker_config.json")
        print("✅ SmartChunkerAdapter инициализирован успешно")

        # Проверяем доступность
        if not adapter.is_available():
            print("❌ SmartChunker недоступен")
            return False

        print("✅ SmartChunker доступен")

        # Тестируем обработку документа
        test_file = "data/input/План строительства моста.docx"
        if Path(test_file).exists():
            print(f"📄 Обрабатываем документ: {test_file}")

            document = adapter.process_document(test_file)
            print(f"✅ Документ обработан успешно!")
            print(f"   - Создано чанков: {len(document.chunks)}")
            print(f"   - Метаданные: {document.metadata}")

            # Показываем первые несколько чанков
            for i, chunk in enumerate(document.chunks[:3]):
                print(f"   Чанк {i+1}:")
                print(f"     - Размер: {len(chunk.content)} символов")
                print(f"     - Секция: {chunk.metadata.get('section_number', 'N/A')}")
                print(f"     - Уровень: {chunk.metadata.get('level', 'N/A')}")
                print(f"     - Начало: {chunk.content[:100]}...")
                print()

            return True
        else:
            print(f"❌ Тестовый файл не найден: {test_file}")
            return False

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_smart_chunker()
    if success:
        print("🎉 Тест прошел успешно!")
    else:
        print("💥 Тест не прошел!")
        sys.exit(1)
