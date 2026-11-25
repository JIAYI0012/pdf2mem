# PDF to Agent Memory

Turn research paper PDFs into evidence-linked, machine-readable memories for agents. The workflow reads the PDF text layer, splits it into page-local chunks, and asks a language model to extract two complementary kinds of memory:

- **Knowledge:** observations, principles, proposed hypotheses, mechanisms, structure–property relationships, and limitations.
- **Experiment:** procedures, characterization, measurements, and results, including partial experimental details where available.

Every accepted memory includes a quotation that must occur verbatim in the extracted page text, together with a document ID, physical page number, chunk ID, and character offsets. The quote provides a way to locate evidence; it does **not** prove that the model's statement or every structured detail is correct. This is a research aid, not an automated scientific fact checker.

## Quick start

Requirements: Python 3.10 or newer, PDFs with a text layer, and an OpenAI-compatible chat-completions API for extraction. The default model is `gpt-4o-mini`; choose a model available from your provider. Extraction sends paper text to that provider, so check that you are permitted to share the documents.

```bash
git clone https://github.com/JIAYI0012/pdf2mem.git
cd pdf2mem
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

export OPENAI_API_KEY='your-api-key'
# Optional: an OpenAI-compatible API base URL ending in /v1.
# export OPENAI_BASE_URL='https://your-provider.example/v1'

pdf-memory papers/ --output output/
```

On Windows, activate with `.venv\Scripts\activate` and set the environment variable using your shell's syntax. A directory input is searched recursively for `.pdf` files; a single PDF path works too. The program writes a JSON summary to standard output. API requests, model availability, and billing depend on the selected provider.

Inspect extracted text first without an API key or model calls:

```bash
pdf-memory paper.pdf --output parsed/ --parse-only
```

Useful options:

```bash
pdf-memory paper.pdf -o output/ --type knowledge --model gpt-4o-mini
pdf-memory papers/ -o output/ --type experiment --chunk-size 10000 --chunk-overlap 1000
pdf-memory --help
```

`--type` accepts `both` (default), `knowledge`, or `experiment`. Chunk sizes and overlap are measured in characters; chunks never cross page boundaries. The overlap must be nonnegative and smaller than the chunk size. The CLI exits with code 0 for a complete or parse-only run, 1 for a partial run with recorded errors, and 2 for invalid input or setup errors.

## Results

Each run writes UTF-8 JSON Lines files to its own `output/runs/<run-id>/` snapshot. The printed summary includes `run_dir` with the exact path. Each line is a JSON object, so agents and scripts can stream records independently. Only selected memory types are emitted; parse-only runs contain parsed artifacts without model extraction.

| File | Contents |
| --- | --- |
| `knowledge.jsonl` | Knowledge memories from this invocation. |
| `experiment.jsonl` | Experimental memories from this invocation. |
| `documents.jsonl` | Document ID, filename, available PDF metadata, empty physical page numbers, and chunk count. |
| `chunks.jsonl` | Extracted chunk text, page, and page-relative character offsets. |
| `errors.jsonl` | PDF or per-chunk extraction failures, with available document, page and chunk context. |
| `manifest.jsonl` | One summary record with counts, status, model, and selected types. |
| `output/cache.sqlite3` | Shared, resumable successful model responses across runs; see below. |

Memory records have this common structure (values are illustrative):

```json
{
  "schema_version": "1.0",
  "memory_id": "<sha256>",
  "memory_type": "knowledge",
  "subtype": "hypothesis",
  "statement": "The authors propose that the ligand stabilizes the cluster.",
  "details": {"entities": ["ligand", "cluster"], "conditions": "", "attribution": "authors"},
  "source": {
    "document_id": "<sha256-of-pdf-bytes>",
    "file_name": "paper.pdf",
    "page": 3,
    "chunk_id": "p3-...",
    "page_char_start": 120,
    "page_char_end": 157,
    "quote": "The ligand may stabilize the cluster."
  },
  "extraction": {"model": "gpt-4o-mini"}
}
```

Character offsets are zero-based in the extracted text of the physical PDF page, with an exclusive end; they are **not** coordinates in the PDF layout. Knowledge `details` is a model-produced object intended to contain `entities`, `conditions`, and `attribution`. Experiment `details` has `objective`, `sample_or_product`, `reagents` (name, amount, role), `steps`, `conditions`, `characterization`, `results`, `yield`, and `missing_information`. Unstated scalar values are represented as `null` and lists as `[]`. The missing-information field refers to the current excerpt, not necessarily the whole paper. Refer to `src/pdf_to_agent_memory/experiment.py` for the validated experiment schema and allowed subtypes.

A document ID is a SHA-256 hash of the PDF bytes, so an identical file is processed once per run even if it appears under multiple names. Chunk IDs reflect the document, page, offsets, and text. Memory IDs are deterministic for the same document, location, quote, type, subtype, and statement. Exact duplicate IDs are removed within a run; semantically similar memories or different model wordings may remain.

## Resume and errors

Successful chunk/type model responses are cached in `output/cache.sqlite3`. Rerun the same command with the same output directory to reuse successful calls and retry failed chunks. Cache keys include document and chunk identity, memory type, model, client endpoint and prompt. Changes to the PDF, chunking, prompt, model, or endpoint may cause new calls. The manifest reports `model_calls` and `cached_calls`; inspect `errors.jsonl` and `documents.jsonl` for errors and empty pages. A failure can coexist with successfully extracted memories, and a partial run returns exit code 1.

**Each invocation creates a separate run snapshot** under `output/runs/<run-id>/`; its JSONL files reflect that invocation’s input and selected types, not an append-only collection. `--parse-only` and single-type runs leave previous snapshots intact. The cache remains in the shared output directory. Avoid running multiple processes against the same output directory at the same time.

## Python and agent integration

```python
from pdf_to_agent_memory import iter_memories, run

summary = run("papers/", "output/", model="gpt-4o-mini",
              types=("knowledge", "experiment"))
print(summary)
for memory in iter_memories(summary["run_dir"], "knowledge"):
    print(memory["statement"], memory["source"]["quote"])
```

A custom client may be injected. It must implement `complete(*, model, system, text) -> dict` and return the response object expected by the prompt. For example, this no-network stub returns no memories and is useful for exercising an integration:

```python
from pdf_to_agent_memory import run

class EmptyClient:
    def complete(self, *, model, system, text):
        return {"memories": []}

summary = run("paper.pdf", "stub-output/", client=EmptyClient(), model="stub")
```

Replace the stub with your own provider adapter. The experiment response must conform to the schema embedded in its system prompt, and each evidence quotation must be a nonempty exact substring of `text`; malformed output or an unmatched quotation fails the chunk/type and appears in `errors.jsonl`. The custom client's class name is part of its cache identity unless it provides a `base_url` attribute; use a distinct output directory or endpoint identity when changing the underlying client behavior. See [examples/custom_client.py](examples/custom_client.py) for a runnable stub.

For extraction without an API call, `run("paper.pdf", "parsed/", parse_only=True)` provides the same parsing-only mode as the CLI.

## Scientific and operational limits

- Only the existing PDF text layer is read. There is no OCR or specialized figure, table, or layout interpretation. Scans may fail or have empty pages; multi-column reading order and symbols may be wrong. Supplementary files are not automatically connected to a paper.
- Models can miss material, misinterpret attribution or uncertainty, or produce plausible but incorrect details. Exact quote matching checks that a quote exists in the extracted text, not that it supports the complete claim. Review the paper and evidence before using memories in scientific decisions or experiments.
- Page-local extraction can split a procedure or argument across chunks; a memory can be incomplete. Overlapping chunks can produce semantically duplicate results. No semantic deduplication, retrieval index, or experiment execution is included.
- The workflow validates the experiment response structure and basic knowledge fields, but cannot establish scientific correctness, units, or whether an experiment can be reproduced. Existing PDF metadata can be incomplete or wrong.
- Protect output files: chunks and quotes can contain copyrighted or confidential paper text. Consult your provider's data handling and access controls before processing sensitive material.

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest
```

## 中文快速使用

本项目从论文 PDF 提取两类可供 agent 使用的记忆：**knowledge**（知识、原理、假设、机理等）和 **experiment**（实验步骤、表征、测量、结果等）。每条记录保留可在抽取文本中定位的原文、页码及偏移量。

安装 Python 3.10 及以上，进入本项目目录后执行：

```bash
python -m pip install -e .
export OPENAI_API_KEY='你的密钥'
pdf-memory 论文目录/ -o 结果目录/
```

可将论文目录换成单个 PDF。仅检查 PDF 文本、无需调用模型时使用 `pdf-memory 论文.pdf -o 解析结果/ --parse-only`。每次运行在 `结果目录/runs/<run-id>/` 中生成独立结果，摘要的 `run_dir` 给出具体路径；其中 `knowledge.jsonl` 和 `experiment.jsonl` 为抽取结果，`errors.jsonl` 为处理失败信息。使用同一个输出目录重跑可复用成功的调用并重试失败项，之前的运行结果会保留。

程序不支持 OCR，模型生成内容需要对照论文人工核实。原文匹配只能证明引用存在，不能证明结论正确；调用模型时论文文本会发送给所配置的 API 服务商。
