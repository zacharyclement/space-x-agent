# SpaceX Agent

FastAPI + LangChain conversational agent that answers SpaceX questions using live SpaceX API data.

## Architecture

- `api/`: FastAPI transport layer and dependency wiring.
- `agent/`: LangChain agent factory, middleware, and state schema.
- `tools/`: Dependency-injected tool definitions and parsing logic.
- `services/`: SpaceX client interface and HTTP implementation.
- `core/`: Settings, logging, and domain exceptions.
- `tests/`: Deterministic unit and API tests.
- `web/`: Minimal HTML chat interface.

## Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:
   - `pip install -e .`
   - `pip install -e .[dev]` (optional for lint/test tooling)
3. Create env file from template:
   - `cp env.example .env`
4. Set at least:
   - `OPENAI_API_KEY`
   - `LANGCHAIN_API_KEY`

> Note: In this workspace runtime, creating hidden files via tooling may be restricted.
> If needed, copy `env.example` to `.env` manually.

## Run

- API + UI:
  - `uvicorn api.main:app --reload`
- Open:
  - `http://127.0.0.1:8000/`

## Memory implementation

Short-term memory is implemented with LangGraph `InMemorySaver` in `agent/factory.py`.

- Conversation state persists per `thread_id`.
- API accepts optional `thread_id`; if omitted, server creates one.
- UI stores `thread_id` in browser `localStorage`.
- Message history is trimmed by middleware (`agent/middleware.py`) to reduce context growth.

## Documentation index reference

Fetch the complete LangChain docs index at:

- `https://docs.langchain.com/llms.txt`

Use it to discover additional pages (including production-grade checkpointer options like Postgres).

## Tests and checks

- `pytest`
- `ruff check .`
- `black --check .`
- `isort --check-only .`
- `mypy .`

## Production memory upgrade path

For production persistence, replace `InMemorySaver` with a database-backed checkpointer (for example Postgres via `langgraph-checkpoint-postgres`) while keeping the same `thread_id` contract.
