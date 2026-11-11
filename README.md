# LLM Guard Proxy

Конфигурируемый защитный слой для LLM-приложений: защита от prompt-инъекций и
соблюдение настраиваемых политик использования. Курсовой проект по дисциплине
«Глубокое обучение», МГТУ им. Н.Э. Баумана.

> **Для ИИ-ассистента:** полный контекст проекта и правила работы — в `CLAUDE.md`.

> Статус: этап 1 (базовый сервис без защиты). План работ — `PLAN.txt`, журнал — `docs/progress.md`.

## Структура

```
config/            настройки и политики (YAML)
src/llm_guard/     код защитного слоя
  api/             HTTP API (FastAPI)
  core/            конвейер, контекст запроса, решения, реестр проверок
  normalize/       нормализация и декодирование текста
  detectors/       детекторы prompt-инъекций
  policies/        типы политик
  output/          проверки ответа модели
  backends/        клиент к LLM
  prompts/         шаблоны промптов
ui/                веб-интерфейс для демонстрации
data/              датасеты (raw — исходные, benchmark — тестовый набор)
tests/             unit, integration, benchmark
reports/           результаты прогонов
docs/              обзор литературы, журнал, записка
scripts/           вспомогательные скрипты
```

## Установка (Windows)

Нужен Python 3.10+.

```
cd /d D:\BMSTU\2_Grade\Deep_learning_KP
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

Скрипт создаст `.venv`, установит зависимости и прогонит тесты.

## Запуск (этап 1)

1. Установить Ollama: https://ollama.com/download и скачать модель:
   `ollama pull qwen3.5:4b`
2. Запустить сервис: `.venv\Scripts\python -m llm_guard`
3. Отправить вопрос (в другом окне PowerShell):

```powershell
$body = @{ messages = @(@{ role = "user"; content = "Привет! Что ты умеешь?" }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/chat/completions -Method Post `
  -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

Тесты: `.venv\Scripts\python -m pytest -q`

## Замеры производительности (ноутбуки)

Один раз установить зависимости: `.venv\Scripts\python -m pip install -e ".[notebook]"`

- `notebooks/01_latency.ipynb` — насколько сервис замедляет ответ модели;
- `notebooks/02_load_test.ipynb` — работа при нескольких одновременных запросах.

Перед запуском: Ollama и сервис запущены. Результаты сохраняются в `reports/`.
