"""Single-process local API and static study app."""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.contracts import DEFAULT_DB, PROJECT_ROOT, SearchRequest, UDLException
from src.providers import ProviderConfig, public_presets


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=6000)
    top_k: int = Field(default=6, ge=1, le=30)
    mode: str = "hybrid"
    modality: str = "all"
    chapter: Optional[str] = None

    def search_request(self):
        if self.mode not in {"lexical", "semantic", "hybrid"}:
            raise UDLException("mode_invalid", "Choose lexical, semantic, or hybrid search.")
        if self.modality not in {"all", "text", "code", "figure", "exercise", "equation"}:
            raise UDLException("modality_invalid", "Choose all, text, code, figure, exercise, or equation.")
        if not self.query.strip():
            raise UDLException("query_empty", "Enter a question or search phrase.")
        return SearchRequest(**self.model_dump(include=set(SearchRequest.__dataclass_fields__)))


class AskInput(SearchInput):
    provider: str
    model: str
    api_key: Optional[str] = Field(default=None, repr=False)
    base_url: Optional[str] = None
    history: list = Field(default_factory=list, max_length=20)


class BuildInput(BaseModel):
    semantic: bool = False
    offline: bool = False


class ReviewInput(BaseModel):
    transcription: str = Field(max_length=30000)
    reviewer: str = Field(min_length=1, max_length=200)
    decision: str
    build_id: Optional[str] = None


def create_app(db_path=DEFAULT_DB, project_root=PROJECT_ROOT, runtime=None):
    from src.runtime import Runtime
    runtime = runtime or Runtime(db_path, project_root)
    root = Path(project_root).resolve()
    app = FastAPI(title="Understanding Deep Learning Tutor", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
    app.state.runtime = runtime

    @app.middleware("http")
    async def local_requests(request, call_next):
        origin = request.headers.get("origin")
        if origin and request.method not in {"GET", "HEAD", "OPTIONS"}:
            from urllib.parse import urlparse
            if urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"error": {"code": "origin_invalid", "message": "Cross-origin changes are not allowed.", "retryable": False}}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; connect-src 'self'; frame-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'self'"
        return response

    @app.exception_handler(UDLException)
    async def udl_error(request, exc):
        status = 404 if exc.code.endswith("not_found") else 409 if exc.code in {"build_busy", "index_empty", "dense_missing", "dense_stale"} else 400
        return JSONResponse({"error": exc.as_dict()}, status_code=status)

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def bad_input(request, exc):
        # Pydantic errors can echo input (including API keys). Never return them.
        return JSONResponse({"error": {"code": "request_invalid", "message": "Check the request fields, question length, and result count (1–30).", "retryable": False}}, status_code=422)

    @app.get("/api/status")
    def status():
        return dict(runtime.status(), providers=public_presets())

    @app.post("/api/search")
    def search(payload: SearchInput):
        request = payload.search_request()
        return runtime.service(request.mode).search(request).as_dict()

    @app.post("/api/ask")
    def ask(payload: AskInput):
        request = payload.search_request()
        config = ProviderConfig(payload.provider, payload.model, payload.api_key, payload.base_url)

        def events():
            try:
                for event, data in runtime.service(request.mode).answer_events(request, config, payload.history):
                    yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            except UDLException as exc:
                yield f"event: error\ndata: {json.dumps(exc.as_dict())}\n\n"
            except Exception:
                yield 'event: error\ndata: {"code":"answer_failed","message":"The answer could not be completed. Check index status and provider settings.","retryable":true}\n\n'
        return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.post("/api/build")
    def build(payload: BuildInput):
        return runtime.start_build(payload.semantic, payload.offline)

    @app.get("/api/chunks/{chunk_id}")
    def chunk(chunk_id: str, build_id: Optional[str] = None):
        result = runtime.store.get_chunk(chunk_id, build_id=build_id)
        if not result:
            raise UDLException("chunk_not_found", "This source chunk is unavailable in the selected build.")
        return result

    def allowed_file(path, parents):
        target = Path(path)
        target = (root / target).resolve() if not target.is_absolute() else target.resolve()
        if not any(target.is_relative_to(parent) for parent in parents) or not target.is_file():
            raise UDLException("asset_not_found", "The original source asset is missing or outside the source directory.")
        return target

    @app.get("/api/source")
    def source(source_id: str, build_id: Optional[str] = None):
        record = runtime.store.get_source(source_id, build_id=build_id)
        if not record:
            raise UDLException("source_not_found", "This source is not part of the selected corpus build.")
        target = allowed_file(record["path"], [(root / "data/raw").resolve(), (root / "data/snapshots").resolve()])
        media = "application/pdf" if target.suffix.lower() == ".pdf" else "application/json"
        return FileResponse(target, media_type=media, headers={"Content-Disposition": "inline"})

    @app.get("/api/asset")
    def asset(chunk_id: str, build_id: Optional[str] = None):
        record = runtime.store.get_chunk(chunk_id, build_id=build_id)
        if not record:
            raise UDLException("chunk_not_found", "This source chunk is unavailable.")
        path = (record.get("metadata") or {}).get("asset_path")
        if not path:
            raise UDLException("asset_not_found", "This chunk has no source crop.")
        target = allowed_file(path, [(root / "data/assets").resolve()])
        return FileResponse(target, media_type="image/png")

    @app.get("/api/reviews")
    def reviews(build_id: Optional[str] = None):
        return {"reviews": runtime.store.list_reviews(build_id=build_id)}

    @app.post("/api/reviews/{chunk_id}")
    def review(chunk_id: str, payload: ReviewInput):
        if payload.decision not in {"verified", "pending"}:
            raise UDLException("review_invalid", "Choose verified or pending.")
        return runtime.store.review_equation(chunk_id, payload.transcription, payload.reviewer, payload.decision, build_id=payload.build_id)

    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="study")
    return app
