# Руководство по разработке

## Правила и стандарты кодирования

### 1. Форматирование кода

- **Black**: Автоматическое форматирование кода
- **isort**: Сортировка импортов
- **Максимальная длина строки**: 88 символов

```bash
# Автоматическое форматирование
black src/ tests/
isort src/ tests/
```

### 2. Линтинг и проверки

- **Flake8**: Проверка стиля кода
- **MyPy**: Проверка типов
- **Bandit**: Проверка безопасности

```bash
# Проверка качества кода
flake8 src/ tests/
mypy src/
bandit -r src/
```

### 3. Тестирование

- **Покрытие кода**: минимум 80%
- **Все тесты должны проходить**
- **Используйте pytest**

```bash
# Запуск тестов
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 4. Pre-commit hooks

Автоматические проверки перед коммитом:

```bash
# Установка pre-commit hooks
pre-commit install
```

### 5. Структура коммитов

Используйте conventional commits:

```
feat: добавить новую функцию
fix: исправить баг
docs: обновить документацию
test: добавить тесты
refactor: рефакторинг кода
```

### 6. Рабочий процесс

1. **Создайте ветку** для новой функции
2. **Напишите код** следуя стандартам
3. **Добавьте тесты** для новой функциональности
4. **Запустите проверки** качества кода
5. **Создайте Pull Request**

### 7. Команды для разработки

```bash
# Полная проверка качества
make check-quality

# Автоматическое исправление форматирования
make fix-formatting

# Запуск всех тестов
make test

# Установка зависимостей для разработки
make install-dev

# Настройка pre-commit hooks
make setup-pre-commit
```

### 8. Требования к коду

- **Типизация**: Используйте type hints
- **Документация**: Добавляйте docstrings
- **Обработка ошибок**: Используйте try/except
- **Логирование**: Используйте logging вместо print
- **Конфигурация**: Используйте .env файлы

### 9. Архитектурные принципы

- **Модульность**: Разделяйте код на модули
- **Тестируемость**: Код должен быть легко тестируемым
- **Расширяемость**: Легко добавлять новые функции
- **Читаемость**: Код должен быть понятным

### 10. Проверки перед коммитом

```bash
# Автоматическая проверка
python scripts/check_code_quality.py --check

# Автоматическое исправление
python scripts/check_code_quality.py --fix
```

## Настройка среды разработки

1. **Установите зависимости**:
   ```bash
   make install-dev
   ```

2. **Настройте pre-commit hooks**:
   ```bash
   make setup-pre-commit
   ```

3. **Проверьте настройку**:
   ```bash
   make check-quality
   ```

## Интеграция с IDE

### VS Code

Создайте `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### PyCharm

1. Настройте Black как форматтер
2. Включите Flake8 и MyPy
3. Настройте автоматическое форматирование при сохранении

## Troubleshooting

### Проблемы с pre-commit

```bash
# Обновить hooks
pre-commit autoupdate

# Запустить вручную
pre-commit run --all-files
```

### Проблемы с MyPy

```bash
# Игнорировать отсутствующие импорты
mypy src/ --ignore-missing-imports
```

### Проблемы с тестами

```bash
# Запустить конкретный тест
pytest tests/test_specific.py::test_function -v

# Запустить с отладкой
pytest tests/ -v -s --pdb
```
