"""Command-line interface."""
import argparse
import json
import sys

from .client import PROVIDERS
from .workflow import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract agent memories from one PDF or a directory of PDFs.")
    parser.add_argument("input", help="PDF or directory (directories are searched recursively)")
    parser.add_argument("--output", "-o", default="memory-output", help="Output directory (default: memory-output)")
    parser.add_argument("--provider", choices=PROVIDERS, help="Override PDF2MEM_PROVIDER for this run")
    parser.add_argument("--model", help="Override the configured provider's chat model")
    parser.add_argument("--env-file", help="Deployment settings file (default: .env in working directory)")
    parser.add_argument("--type", choices=["both", "knowledge", "experiment"], default="both")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--chunk-overlap", type=int, default=1000)
    parser.add_argument("--parse-only", action="store_true", help="Extract text and chunks without API calls")
    args = parser.parse_args()
    try:
        types = ("knowledge", "experiment") if args.type == "both" else (args.type,)
        def progress(event):
            print(f"{event['file_name']} page {event['page']} {event['type']}: {event['status']}", file=sys.stderr)
        summary = run(args.input, args.output, model=args.model, provider=args.provider, env_file=args.env_file, types=types,
                      chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, parse_only=args.parse_only,
                      on_progress=progress)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
