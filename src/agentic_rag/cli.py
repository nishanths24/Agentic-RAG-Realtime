from __future__ import annotations

import argparse
import json
import sys

from agentic_rag.logging_setup import setup_logging
from agentic_rag.runtime import build_orchestrator, get_settings
from agentic_rag.tools.registry import ToolRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-rag", description="Agentic RAG with real-time tools")
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="Build or refresh the vector index")
    p_index.add_argument("--rebuild", action="store_true")
    sub.add_parser("index-stats", help="Show index counts")

    p_search = sub.add_parser("search", help="Semantic search over the index")
    p_search.add_argument("query")

    p_ask = sub.add_parser("ask", help="Ask a grounded question")
    p_ask.add_argument("question")
    p_ask.add_argument("--trace", action="store_true")

    p_tool = sub.add_parser("tool", help="Run a single tool")
    p_tool.add_argument("name")
    p_tool.add_argument("--timezone")
    p_tool.add_argument("--expression")
    p_tool.add_argument("--amount", type=float)
    p_tool.add_argument("--from")
    p_tool.add_argument("--to")
    p_tool.add_argument("--city")
    p_tool.add_argument("--args", default="", help="JSON object of arguments")

    p_serve = sub.add_parser("serve", help="Run the API and UI")
    p_serve.add_argument("--host")
    p_serve.add_argument("--port", type=int)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_dir)

    if args.command == "serve":
        import uvicorn

        host = args.host or settings.host
        port = args.port or settings.port
        uvicorn.run("agentic_rag.api.app:app", host=host, port=port, reload=False)
        return 0

    orchestrator, indexer, store = build_orchestrator(settings)

    if args.command == "index":
        print(json.dumps(indexer.build(rebuild=args.rebuild), indent=2))
        return 0
    if args.command == "index-stats":
        print(json.dumps({"chunks": store.count(), "documents": store.document_hashes()}, indent=2))
        return 0
    if args.command == "search":
        hits = orchestrator.retriever.search(
            args.query, top_k=settings.top_k, min_score=settings.min_score
        )
        for hit in hits:
            preview = hit.chunk.text.replace("\n", " ")[:180]
            print(f"{hit.rank}. {hit.score:.3f}  {hit.chunk.source_path}  {preview}")
        if not hits:
            print("No matching passages.")
        return 0
    if args.command == "ask":
        answer = orchestrator.ask(args.question)
        print(answer.answer)
        print("\nFrom your documents:")
        for cite in answer.document_evidence:
            print(f"  [{cite.label}] {cite.reference}")
        print("Live information:")
        for cite in answer.live_evidence:
            print(f"  [{cite.label}] {cite.reference}")
        if answer.caveats:
            print("Caveats:")
            for caveat in answer.caveats:
                print(f"  - {caveat}")
        if args.trace:
            print("\nTrace:")
            for step in answer.trace:
                print(f"  {step.name}: {step.detail} ({step.duration_ms:.0f} ms)")
        return 0
    if args.command == "tool":
        arguments = _tool_args(args)
        result = ToolRegistry().run(args.name, arguments, timeout_s=settings.request_timeout_s)
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "data": result.data,
                    "error": result.error,
                    "source": result.source,
                    "retrieved_at": result.retrieved_at.isoformat(),
                },
                indent=2,
            )
        )
        return 0 if result.ok else 2
    return 1


def _tool_args(args: argparse.Namespace) -> dict:
    if args.args:
        return json.loads(args.args)
    from_code = getattr(args, "from")
    mapping = {
        "current_time": {"timezone": args.timezone or "Asia/Kolkata"},
        "calculator": {"expression": args.expression or "1+1"},
        "currency_convert": {
            "amount": args.amount or 1,
            "from_currency": from_code or "USD",
            "to_currency": args.to or "INR",
        },
        "weather": {"city": args.city or "Paris"},
    }
    if args.name not in mapping:
        raise SystemExit(f"Pass --args JSON for tool {args.name}")
    return mapping[args.name]


if __name__ == "__main__":
    sys.exit(main())
