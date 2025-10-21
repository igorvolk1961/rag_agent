"""
Command Line Interface for RAG Agent
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings
from core.rag_engine import RAGEngine
from core.document_processor import DocumentProcessor
from utils.logging_utils import setup_logging


console = Console()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Включить подробный вывод')
@click.option('--config', '-c', type=click.Path(exists=True), help='Путь к файлу конфигурации')
@click.pass_context
def cli(ctx, verbose, config):
    """RAG Agent - Система поиска и генерации ответов на основе документов"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config

    # Настройка логирования
    if verbose:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=logging.INFO)


@cli.command()
@click.option('--file', '-f', 'files', multiple=True, help='Файлы для добавления')
@click.option('--directory', '-d', type=click.Path(exists=True), help='Директория с файлами')
@click.option('--recursive', '-r', is_flag=True, help='Рекурсивный поиск в директории')
@click.pass_context
def add_documents(ctx, files, directory, recursive):
    """Добавить документы в RAG систему"""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Инициализация RAG системы...", total=None)

            # Загрузка настроек
            settings = Settings()
            rag_engine = RAGEngine(settings)
            document_processor = DocumentProcessor(settings)

            progress.update(task, description="Обработка документов...")

            # Сбор файлов для обработки
            files_to_process = []

            if files:
                files_to_process.extend(files)

            if directory:
                dir_path = Path(directory)
                if recursive:
                    pattern = "**/*"
                else:
                    pattern = "*"

                for file_path in dir_path.glob(pattern):
                    if file_path.is_file() and document_processor.is_supported_file(str(file_path)):
                        files_to_process.append(str(file_path))

            if not files_to_process:
                console.print("[red]Ошибка: Не найдено файлов для обработки[/red]")
                return

            # Обработка файлов
            total_files = len(files_to_process)
            progress.update(task, total=total_files, description=f"Обработка {total_files} файлов...")

            processed_count = 0
            for file_path in files_to_process:
                try:
                    progress.update(task, description=f"Обработка: {Path(file_path).name}")

                    # Обработка и добавление документа
                    chunks = document_processor.process_and_chunk(file_path)
                    if chunks:
                        rag_engine.add_documents(chunks)
                        processed_count += 1

                    progress.advance(task)

                except Exception as e:
                    console.print(f"[yellow]Предупреждение: Ошибка при обработке {file_path}: {e}[/yellow]")
                    progress.advance(task)

            progress.update(task, description="Завершение...")

        # Вывод результатов
        console.print(f"\n[green]✅ Успешно обработано {processed_count} из {total_files} файлов[/green]")

        # Статистика
        stats = rag_engine.get_stats()
        table = Table(title="Статистика RAG системы")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="green")

        table.add_row("Документов", str(stats.get('documents', 0)))
        table.add_row("Чанков", str(stats.get('chunks', 0)))
        table.add_row("Размер векторного хранилища", str(stats.get('vector_store_size', 0)))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Ошибка: {e}[/red]")
        if ctx.obj['verbose']:
            console.print_exception()


@cli.command()
@click.argument('query', nargs=-1)
@click.option('--interactive', '-i', is_flag=True, help='Интерактивный режим')
@click.option('--stream', '-s', is_flag=True, help='Потоковый вывод ответа')
@click.pass_context
def query(ctx, query, interactive, stream):
    """Задать вопрос RAG системе"""
    try:
        # Инициализация
        settings = Settings()
        rag_engine = RAGEngine(settings)

        if interactive:
            # Интерактивный режим
            console.print(Panel.fit(
                "[bold blue]RAG Agent - Интерактивный режим[/bold blue]\n"
                "Введите 'exit' или 'quit' для выхода",
                title="🤖 RAG Agent"
            ))

            while True:
                user_query = Prompt.ask("\n[cyan]Ваш вопрос[/cyan]")

                if user_query.lower() in ['exit', 'quit', 'выход']:
                    console.print("[yellow]До свидания![/yellow]")
                    break

                if not user_query.strip():
                    continue

                try:
                    if stream:
                        console.print("[blue]Ответ:[/blue]")
                        for chunk in rag_engine.stream_query(user_query):
                            console.print(chunk, end="")
                        console.print()  # Новая строка
                    else:
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=console
                        ) as progress:
                            task = progress.add_task("Обработка запроса...", total=None)
                            answer = rag_engine.query(user_query)
                            progress.stop()

                        console.print(f"[blue]Ответ:[/blue] {answer}")

                except Exception as e:
                    console.print(f"[red]Ошибка при обработке запроса: {e}[/red]")

        else:
            # Одноразовый запрос
            if not query:
                console.print("[red]Ошибка: Не указан запрос[/red]")
                return

            user_query = " ".join(query)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Обработка запроса...", total=None)

                if stream:
                    progress.stop()
                    console.print(f"[cyan]Запрос:[/cyan] {user_query}")
                    console.print("[blue]Ответ:[/blue]")
                    for chunk in rag_engine.stream_query(user_query):
                        console.print(chunk, end="")
                    console.print()  # Новая строка
                else:
                    answer = rag_engine.query(user_query)
                    progress.stop()

                    console.print(f"[cyan]Запрос:[/cyan] {user_query}")
                    console.print(f"[blue]Ответ:[/blue] {answer}")

    except Exception as e:
        console.print(f"[red]Ошибка: {e}[/red]")
        if ctx.obj['verbose']:
            console.print_exception()


@cli.command()
@click.pass_context
def stats(ctx):
    """Показать статистику RAG системы"""
    try:
        settings = Settings()
        rag_engine = RAGEngine(settings)

        stats = rag_engine.get_stats()

        # Создание таблицы статистики
        table = Table(title="📊 Статистика RAG системы")
        table.add_column("Параметр", style="cyan", no_wrap=True)
        table.add_column("Значение", style="green")
        table.add_column("Описание", style="dim")

        table.add_row(
            "Документов",
            str(stats.get('documents', 0)),
            "Количество обработанных документов"
        )
        table.add_row(
            "Чанков",
            str(stats.get('chunks', 0)),
            "Количество текстовых фрагментов"
        )
        table.add_row(
            "Размер векторного хранилища",
            str(stats.get('vector_store_size', 0)),
            "Количество векторов в хранилище"
        )
        table.add_row(
            "Тип векторного хранилища",
            settings.vector_store_type,
            "Используемое хранилище векторов"
        )
        table.add_row(
            "Модель LLM",
            settings.llm_model,
            "Используемая языковая модель"
        )
        table.add_row(
            "Тип эмбеддингов",
            settings.embedding_type,
            "Модель для создания эмбеддингов"
        )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Ошибка: {e}[/red]")
        if ctx.obj['verbose']:
            console.print_exception()


@cli.command()
@click.option('--confirm', is_flag=True, help='Подтвердить очистку без запроса')
@click.pass_context
def clear(ctx, confirm):
    """Очистить все документы из RAG системы"""
    try:
        if not confirm:
            if not Confirm.ask("Вы уверены, что хотите очистить все документы?"):
                console.print("[yellow]Операция отменена[/yellow]")
                return

        settings = Settings()
        rag_engine = RAGEngine(settings)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Очистка RAG системы...", total=None)
            rag_engine.clear_knowledge_base()
            progress.stop()

        console.print("[green]✅ RAG система успешно очищена[/green]")

    except Exception as e:
        console.print(f"[red]Ошибка: {e}[/red]")
        if ctx.obj['verbose']:
            console.print_exception()


@cli.command()
@click.pass_context
def config(ctx):
    """Показать текущую конфигурацию"""
    try:
        settings = Settings()

        # Создание таблицы конфигурации
        table = Table(title="⚙️ Конфигурация RAG системы")
        table.add_column("Параметр", style="cyan", no_wrap=True)
        table.add_column("Значение", style="green")
        table.add_column("Описание", style="dim")

        config_items = [
            ("LLM модель", settings.llm_model, "Используемая языковая модель"),
            ("API ключ DeepSeek", "***" + settings.deepseek_api_key[-4:] if settings.deepseek_api_key else "Не установлен", "API ключ для DeepSeek"),
            ("Тип эмбеддингов", settings.embedding_type, "Модель для создания эмбеддингов"),
            ("Тип векторного хранилища", settings.vector_store_type, "Хранилище векторов"),
            ("Название коллекции", settings.collection_name, "Имя коллекции в векторном хранилище"),
            ("Размер чанка", str(settings.chunk_size), "Размер текстового фрагмента"),
            ("Перекрытие чанков", str(settings.chunk_overlap), "Перекрытие между чанками"),
            ("Количество результатов", str(settings.retrieval_k), "Количество релевантных документов"),
            ("Порог схожести", str(settings.similarity_threshold), "Минимальный порог схожести"),
            ("Температура", str(settings.temperature), "Температура генерации"),
            ("Максимум токенов", str(settings.max_tokens) if settings.max_tokens else "По умолчанию", "Максимальное количество токенов"),
            ("Потоковый вывод", "Включен" if settings.use_streaming else "Отключен", "Потоковый вывод ответов"),
            ("Память", "Включена" if settings.enable_memory else "Отключена", "Контекстная память"),
            ("Уровень логирования", settings.log_level, "Уровень детализации логов"),
        ]

        for param, value, description in config_items:
            table.add_row(param, str(value), description)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Ошибка: {e}[/red]")
        if ctx.obj['verbose']:
            console.print_exception()


if __name__ == '__main__':
    cli()
