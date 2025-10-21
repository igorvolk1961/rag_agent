"""
Main entry point for RAG Agent with interactive menu
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings
from core.rag_engine import RAGEngine
from core.document_processor import DocumentProcessor
from utils.logging_utils import setup_logging


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def show_menu():
    """Display the main menu"""
    print("=" * 60)
    print("🤖 RAG AGENT - Система поиска и генерации ответов")
    print("=" * 60)
    print("1. 📄 Добавить документы")
    print("2. ❓ Задать вопрос")
    print("3. 📊 Показать статистику")
    print("4. ⚙️  Показать конфигурацию")
    print("5. 🗑️  Очистить базу знаний")
    print("6. 🔄 Интерактивный режим")
    print("7. ❌ Выход")
    print("=" * 60)


def add_documents_menu(rag_engine, document_processor):
    """Menu for adding documents"""
    print("\n📄 ДОБАВЛЕНИЕ ДОКУМЕНТОВ")
    print("-" * 30)

    while True:
        print("\nВыберите действие:")
        print("1. Добавить файл")
        print("2. Добавить все файлы из папки")
        print("3. Добавить файл с SmartChunker (иерархический чанкинг)")
        print("4. Добавить все файлы из папки с SmartChunker")
        print("5. Назад в главное меню")

        choice = input("\nВаш выбор (1-5): ").strip()

        if choice == "1":
            file_path = input("Введите путь к файлу: ").strip()
            if file_path and Path(file_path).exists():
                try:
                    print("Обработка документа...")
                    chunks_count = rag_engine.add_documents([Path(file_path)])
                    print(f"✅ Документ успешно добавлен! Создано {chunks_count} фрагментов.")
                except Exception as e:
                    print(f"❌ Ошибка: {e}")
            else:
                print("❌ Файл не найден!")

        elif choice == "2":
            folder_path = input("Введите путь к папке: ").strip()
            if folder_path and Path(folder_path).exists():
                try:
                    print("Поиск документов...")
                    folder = Path(folder_path)
                    files = []
                    for file_path in folder.glob("*"):
                        if file_path.is_file() and document_processor.is_supported_file(str(file_path)):
                            files.append(file_path)

                    if files:
                        print(f"Найдено {len(files)} документов. Обработка...")
                        chunks_count = rag_engine.add_documents(files)
                        print(f"✅ Успешно добавлено {len(files)} документов! Создано {chunks_count} фрагментов.")
                    else:
                        print("❌ Документы не найдены в папке.")
                except Exception as e:
                    print(f"❌ Ошибка: {e}")
            else:
                print("❌ Папка не найдена!")

        elif choice == "3":
            # Добавить файл с SmartChunker
            file_path = input("Введите путь к файлу: ").strip()
            if file_path and Path(file_path).exists():
                try:
                    print("Обработка документа с SmartChunker...")
                    chunks_count = rag_engine.add_documents_with_smart_chunker([Path(file_path)])
                    print(f"✅ Документ успешно добавлен с иерархическим чанкингом! Создано {chunks_count} фрагментов.")
                except Exception as e:
                    print(f"❌ Ошибка: {e}")
            else:
                print("❌ Файл не найден!")

        elif choice == "4":
            # Добавить все файлы из папки с SmartChunker
            folder_path = input("Введите путь к папке: ").strip()
            if folder_path and Path(folder_path).exists():
                try:
                    print("Поиск документов...")
                    folder = Path(folder_path)
                    files = []
                    for ext in document_processor.get_supported_extensions():
                        files.extend(folder.glob(f"*{ext}"))

                    if files:
                        print(f"Найдено {len(files)} документов. Обработка с SmartChunker...")
                        chunks_count = rag_engine.add_documents_with_smart_chunker(files)
                        print(f"✅ Все документы успешно добавлены с иерархическим чанкингом! Создано {chunks_count} фрагментов.")
                    else:
                        print("❌ Документы не найдены в папке.")
                except Exception as e:
                    print(f"❌ Ошибка: {e}")
            else:
                print("❌ Папка не найдена!")

        elif choice == "5":
            break
        else:
            print("❌ Неверный выбор!")


def query_menu(rag_engine):
    """Menu for querying the system"""
    print("\n❓ ЗАПРОС К СИСТЕМЕ")
    print("-" * 25)

    while True:
        query = input("\nВведите ваш вопрос (или 'назад' для возврата): ").strip()

        if query.lower() in ['назад', 'back', 'exit']:
            break

        if not query:
            continue

        try:
            print("\n🤔 Обработка запроса...")
            response = rag_engine.query(query)
            print(f"\n💡 Ответ: {response.answer}")

            if response.sources:
                print(f"\n📚 Источники ({len(response.sources)}):")
                for i, source in enumerate(response.sources, 1):
                    print(f"  {i}. {source.get('source', 'Неизвестный источник')}")

        except Exception as e:
            print(f"❌ Ошибка: {e}")


def interactive_mode(rag_engine):
    """Interactive chat mode"""
    print("\n🔄 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("-" * 30)
    print("Введите 'выход' для завершения")
    print("-" * 30)

    while True:
        query = input("\n👤 Вы: ").strip()

        if query.lower() in ['выход', 'exit', 'quit']:
            print("👋 До свидания!")
            break

        if not query:
            continue

        try:
            print("🤖 RAG Agent: ", end="", flush=True)
            for chunk in rag_engine.stream_query(query):
                print(chunk, end="", flush=True)
            print()  # Новая строка

        except Exception as e:
            print(f"❌ Ошибка: {e}")


def show_stats(rag_engine):
    """Show system statistics"""
    print("\n📊 СТАТИСТИКА СИСТЕМЫ")
    print("-" * 25)

    try:
        stats = rag_engine.get_stats()

        print(f"📄 Документов: {stats.get('total_documents', 0)}")
        print(f"📝 Фрагментов: {stats.get('total_chunks', 0)}")
        print(f"🤖 LLM модель: {stats.get('llm_model', 'Неизвестно')}")
        print(f"🔤 Модель эмбеддингов: {stats.get('embedding_model', 'Неизвестно')}")
        print(f"🗄️ Тип хранилища: {stats.get('vector_store_type', 'Неизвестно')}")
        print(f"📏 Размер фрагмента: {stats.get('chunk_size', 'Неизвестно')}")
        print(f"🔄 Перекрытие: {stats.get('chunk_overlap', 'Неизвестно')}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


def show_config():
    """Show system configuration"""
    print("\n⚙️ КОНФИГУРАЦИЯ СИСТЕМЫ")
    print("-" * 30)

    try:
        settings = Settings()

        print(f"🤖 LLM модель: {settings.llm_model}")
        print(f"🔑 API ключ: {'***' + settings.deepseek_api_key[-4:] if settings.deepseek_api_key else 'Не установлен'}")
        print(f"🔤 Тип эмбеддингов: {settings.embedding_type}")
        print(f"🗄️ Тип хранилища: {settings.vector_store_type}")
        print(f"📁 Коллекция: {settings.collection_name}")
        print(f"📏 Размер фрагмента: {settings.chunk_size}")
        print(f"🔄 Перекрытие: {settings.chunk_overlap}")
        print(f"🔍 Количество результатов: {settings.retrieval_k}")
        print(f"📊 Порог схожести: {settings.similarity_threshold}")
        print(f"🌡️ Температура: {settings.temperature}")
        print(f"📝 Макс. токенов: {settings.max_tokens or 'По умолчанию'}")
        print(f"📡 Потоковый вывод: {'Включен' if settings.use_streaming else 'Отключен'}")
        print(f"🧠 Память: {'Включена' if settings.enable_memory else 'Отключена'}")
        print(f"📋 Уровень логов: {settings.log_level}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


def clear_knowledge_base(rag_engine):
    """Clear the knowledge base"""
    print("\n🗑️ ОЧИСТКА БАЗЫ ЗНАНИЙ")
    print("-" * 30)

    confirm = input("⚠️ Вы уверены, что хотите очистить все документы? (да/нет): ").strip().lower()

    if confirm in ['да', 'yes', 'y']:
        try:
            rag_engine.clear_knowledge_base()
            print("✅ База знаний успешно очищена!")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    else:
        print("❌ Операция отменена.")


def main():
    """Main function with interactive menu"""
    try:
        # Setup logging
        setup_logging()

        # Load configuration
        settings = Settings()

        # Initialize components
        rag_engine = RAGEngine(settings)
        document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            use_smart_chunker=getattr(settings, 'use_smart_chunker', False),
            smart_chunker_config=getattr(settings, 'smart_chunker_config_path', None)
        )

        print("🚀 Инициализация RAG системы...")
        print("✅ Система готова к работе!")

        while True:
            clear_screen()
            show_menu()

            choice = input("\nВаш выбор (1-7): ").strip()

            if choice == "1":
                add_documents_menu(rag_engine, document_processor)
            elif choice == "2":
                query_menu(rag_engine)
            elif choice == "3":
                show_stats(rag_engine)
            elif choice == "4":
                show_config()
            elif choice == "5":
                clear_knowledge_base(rag_engine)
            elif choice == "6":
                interactive_mode(rag_engine)
            elif choice == "7":
                print("\n👋 До свидания!")
                break
            else:
                print("❌ Неверный выбор! Нажмите Enter для продолжения...")
                input()

    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем. До свидания!")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
