"""Run the workflow with a no-network client; replace the stub with an adapter.

Usage: python examples/custom_client.py path/to/paper.pdf output-dir
"""
import json
import sys

from pdf_to_agent_memory import run


class EmptyClient:
    def complete(self, *, model, system, text):
        # A real implementation submits system and text to a model and returns
        # its decoded JSON object. Follow the response format in system.
        return {"memories": []}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python examples/custom_client.py PAPER.pdf OUTPUT_DIR")
    summary = run(sys.argv[1], sys.argv[2], client=EmptyClient(), model="stub")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if summary["errors"] else 0)
