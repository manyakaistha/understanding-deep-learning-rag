"""Run with uv run python -m src.cli --help."""

import argparse
import json
import os
import sys
from pathlib import Path

from src.contracts import DEFAULT_DB, PROJECT_ROOT, SearchRequest, UDLException


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def parser():
    main = argparse.ArgumentParser(description="Understanding Deep Learning: handwritten SQLite RAG tutor")
    commands = main.add_subparsers(dest="command", required=True)

    def command(name, help):
        p = commands.add_parser(name, help=help)
        p.add_argument("--db", type=Path, default=DEFAULT_DB)
        return p

    for name in ("build", "update"):
        p = command(name, "Build/update the pinned corpus; activate only after successful indexing")
        p.add_argument("--semantic", action="store_true", help="Download/load the pinned local encoder and build dense vectors")
        p.add_argument("--offline", action="store_true", help="Require existing pinned source files")
    command("status", "Show active snapshot, index readiness, and review gates")
    command("doctor", "Check SQLite integrity and local setup without loading model weights")
    p = command("serve", "Start the local browser tutor")
    p.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost", "::1"])
    p.add_argument("--port", type=int, default=8000)

    for name in ("search", "ask"):
        p = command(name, "Retrieve cited sources" if name == "search" else "Generate a cited answer using your selected provider")
        p.add_argument("question")
        p.add_argument("--top-k", type=int, default=6)
        p.add_argument("--mode", choices=["lexical", "hybrid", "semantic"], default="hybrid")
        p.add_argument("--modality", choices=["all", "text", "code", "figure", "exercise", "equation"], default="all")
        p.add_argument("--chapter")
        if name == "ask":
            p.add_argument("--provider", choices=["groq", "gemini", "openai_compat", "ollama"], default=os.environ.get("UDL_PROVIDER", "ollama"))
            p.add_argument("--model", default=os.environ.get("UDL_MODEL", ""))
            p.add_argument("--base-url")
            p.add_argument("--max-tokens", type=int, default=2048)
            # Keys come from environment, never command-line arguments/history.

    p = command("review", "List pending math or record a source-compared transcription")
    p.add_argument("--chunk-id")
    p.add_argument("--transcription-file", type=Path)
    p.add_argument("--reviewer")
    p.add_argument("--decision", choices=["pending", "verified"], default="pending")
    p.add_argument("--build-id")
    p.add_argument("--pending", action="store_true")

    p = command("evaluate", "Run the fixed real-corpus retrieval benchmark")
    p.add_argument("--suite", type=Path, default=PROJECT_ROOT / "resources/evaluation.json")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/evaluation/latest.json")
    p.add_argument("--mode", choices=["lexical", "hybrid", "semantic"], default="hybrid")
    p.add_argument("--split", choices=["all", "development", "heldout"], default="heldout")
    return main


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from src.runtime import Runtime
        runtime = Runtime(args.db)
        if args.command in {"build", "update"}:
            _print(runtime.build(args.semantic, args.offline))
        elif args.command == "status":
            _print(runtime.status())
        elif args.command == "doctor":
            import sqlite3
            import importlib.util
            with sqlite3.connect(args.db) as db:
                integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            _print(dict(runtime.status(), integrity=integrity,
                        dependencies={name: bool(importlib.util.find_spec(name)) for name in ("torch", "transformers", "fitz", "fastapi", "httpx")},
                        source_book_present=(PROJECT_ROOT / "data/raw/UnderstandingDeepLearning.pdf").is_file()))
            if integrity != "ok":
                return 1
        elif args.command == "serve":
            import uvicorn
            from src.web import create_app
            uvicorn.run(create_app(args.db), host=args.host, port=args.port, access_log=False)
        elif args.command in {"search", "ask"}:
            if not 1 <= args.top_k <= 30 or not args.question.strip():
                raise UDLException("request_invalid", "Provide a question and top-k between 1 and 30.")
            request = SearchRequest(args.question, args.top_k, args.mode, args.modality, args.chapter)
            service = runtime.service(args.mode)
            if args.command == "search":
                _print(service.search(request).as_dict())
            else:
                from src.providers import ProviderConfig
                _print(service.answer(request, ProviderConfig(args.provider, args.model, base_url=args.base_url, max_tokens=args.max_tokens)))
        elif args.command == "review":
            if args.chunk_id:
                if not args.transcription_file or not args.reviewer:
                    raise UDLException("review_invalid", "Recording a review requires --transcription-file and --reviewer.")
                text = args.transcription_file.read_text(encoding="utf-8")
                _print(runtime.store.review_equation(args.chunk_id, text, args.reviewer, args.decision, build_id=args.build_id))
            else:
                reviews = runtime.store.list_reviews(build_id=args.build_id)
                if args.pending:
                    reviews = [r for r in reviews if (r.get("metadata") or {}).get("review_status") != "verified"]
                _print({"reviews": reviews})
        elif args.command == "evaluate":
            from src.evaluation import evaluate
            import hashlib
            report = evaluate(runtime, args.suite, args.mode, args.split)
            report["suite_sha256"] = hashlib.sha256(args.suite.read_bytes()).hexdigest()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            _print(report)
            return 0 if report.get("passed") else 1
    except UDLException as exc:
        _print({"error": exc.as_dict()})
        return 1
    except (FileNotFoundError, ValueError) as exc:
        _print({"error": {"code": "input_error", "message": str(exc), "retryable": False}})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
