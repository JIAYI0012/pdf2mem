"""Exercise provider configuration through the CLI, workflow and HTTP request."""
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pdf_to_agent_memory import run
from pdf_to_agent_memory.cli import main
from test_workflow import FakeClient, make_pdf, rows


def chat_response(request, timeout):
    payload = json.loads(request.data)
    response = FakeClient().complete(
        model=payload["model"],
        system=payload["messages"][0]["content"],
        text=payload["messages"][1]["content"],
    )
    body = {"choices": [{"message": {"content": json.dumps(response)}}]}
    return io.BytesIO(json.dumps(body).encode())


class ProviderTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(temp.name)
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        transport = patch("pdf_to_agent_memory.client.urlopen", side_effect=chat_response)
        self.http = transport.start()
        self.addCleanup(transport.stop)
        make_pdf(Path("paper.pdf"), ["EvidenceK. EvidenceE."])

    def test_dotenv_qwen_both_extractors_use_deployment_credentials(self):
        Path(".env").write_text(
            "PDF2MEM_PROVIDER=qwen\nQWEN_MODEL=qwen-test\n"
            "QWEN_BASE_URL=https://qwen.example/v1\n"
            "DASHSCOPE_API_KEY=file-credential\nOPENAI_API_KEY=unused-credential\n"
        )
        os.environ["DASHSCOPE_API_KEY"] = "process-credential"
        summary = run("paper.pdf", "out")
        self.assertEqual((summary["knowledge"], summary["experiment"]), (1, 1))
        self.assertEqual(self.http.call_count, 2)
        for call in self.http.call_args_list:
            request = call.args[0]
            self.assertEqual(request.full_url, "https://qwen.example/v1/chat/completions")
            self.assertEqual(request.get_header("Authorization"), "Bearer process-credential")
            self.assertEqual(json.loads(request.data)["model"], "qwen-test")
            self.assertIs(json.loads(request.data)["enable_thinking"], False)
        snapshot = Path(summary["run_dir"])
        for kind in ("knowledge", "experiment"):
            record = rows(snapshot, kind)[0]
            self.assertEqual(record["extraction"]["model"], "qwen-test")
            self.assertIn(record["source"]["quote"], "EvidenceK. EvidenceE.")
        for artifact in Path("out").rglob("*"):
            if artifact.is_file():
                self.assertNotIn(b"-credential", artifact.read_bytes(), artifact.name)

    def test_cli_can_select_compatible_provider_and_override_file_model(self):
        Path("deployment.env").write_text(
            "PDF2MEM_PROVIDER=openai\nCOMPATIBLE_BASE_URL=https://gateway.example/v1\n"
            "COMPATIBLE_MODEL=configured-model\nCOMPATIBLE_API_KEY=gateway-credential\n"
        )
        args = ["pdf-memory", "paper.pdf", "--env-file", "deployment.env",
                "--provider", "compatible", "--model", "selected-model", "--type", "knowledge"]
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", args), redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(main(), 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["model"], "selected-model")
        request = self.http.call_args.args[0]
        self.assertEqual(request.full_url, "https://gateway.example/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer gateway-credential")
        self.assertEqual(json.loads(request.data)["model"], "selected-model")
        self.assertNotIn("gateway-credential", stdout.getvalue() + stderr.getvalue())

    def test_legacy_openai_and_provider_specific_cache_at_same_endpoint(self):
        os.environ.update({
            "OPENAI_API_KEY": "openai-credential", "OPENAI_BASE_URL": "https://same.example/v1",
            "QWEN_BASE_URL": "https://same.example/v1", "DASHSCOPE_API_KEY": "qwen-credential",
        })
        legacy = run("paper.pdf", "out", types=("knowledge",))
        self.assertEqual(legacy["model"], "gpt-4o-mini")
        request = self.http.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer openai-credential")
        self.assertEqual(request.full_url, "https://same.example/v1/chat/completions")
        for provider, calls in [("openai", 1), ("qwen", 1), ("qwen", 0)]:
            summary = run("paper.pdf", "out", provider=provider, model="shared-model", types=("knowledge",))
            self.assertEqual(summary["model_calls"], calls)
            self.assertEqual(summary["cached_calls"], 1 - calls)

    def test_missing_selected_key_never_sends_other_providers_key(self):
        os.environ["OPENAI_API_KEY"] = "openai-credential"
        os.environ["QWEN_BASE_URL"] = "https://qwen.example/v1"
        with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
            run("paper.pdf", "out", provider="qwen")
        self.http.assert_not_called()
        parsed = run("paper.pdf", "out", parse_only=True, provider="qwen")
        self.assertEqual(parsed["status"], "parsed")
        self.http.assert_not_called()

    def test_local_compatible_service_without_auth_keeps_external_credentials_local(self):
        os.environ.update({"OPENAI_API_KEY": "openai-credential",
                           "DASHSCOPE_API_KEY": "qwen-credential",
                           "COMPATIBLE_BASE_URL": "http://localhost:8000/v1",
                           "COMPATIBLE_MODEL": "local-model"})
        summary = run("paper.pdf", "out", provider="compatible", types=("knowledge",))
        self.assertEqual(summary["knowledge"], 1)
        request = self.http.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:8000/v1/chat/completions")
        self.assertIsNone(request.get_header("Authorization"))
