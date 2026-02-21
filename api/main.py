"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.dependencies import AppContainer, build_app_container, close_app_container
from api.routes.chat import router as chat_router
from core.settings import get_settings


def create_app(
    *,
    container_builder: Callable[[], AppContainer] = build_app_container,
) -> FastAPI:
    """Create and configure the FastAPI app.

    Args:
        container_builder: Factory used to construct dependency container.

    Returns:
        Configured FastAPI app.
    """
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = container_builder()
        app.state.container = container
        try:
            yield
        finally:
            await close_app_container(container)

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat_router)

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(_index_file_path())

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _index_file_path() -> Path:
    return Path(__file__).resolve().parents[1] / "web" / "index.html"


app = create_app()
