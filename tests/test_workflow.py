"""Offline integration tests for the public workflow and its output artifacts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from pdf_to_agent_memory.workflow import run


def make_pdf(path, texts):
    """Create a small real PDF with a text layer; no network or external binaries."""
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        content = DecodedStreamObject()
        escaped = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        content.set_data(f'BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET'.encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(content)
    with path.open('wb') as output:
        writer.write(output)


def rows(output, kind):
    return [json.loads(line) for line in (output / f'{kind}.jsonl').read_text(encoding='utf-8').splitlines()]


class FakeClient:
    def __init__(self):
        self.calls = []
        self.fail_experiment_once = False
        self.bad_knowledge_once = False

    def complete(self, *, model, system, text):
        kind = 'experiment' if 'Extract experimentally relevant' in system else 'knowledge'
        self.calls.append((kind, text))
        if kind == 'experiment' and self.fail_experiment_once:
            self.fail_experiment_once = False
            raise RuntimeError('temporary failure')
        if kind == 'knowledge':
            if self.bad_knowledge_once:
                self.bad_knowledge_once = False
                return {'memories': [{'subtype': 'hypothesis', 'statement': 'A hypothesis',
                                      'evidence': 'not in the paper', 'details': {'entities': [],
                                      'conditions': '', 'attribution': 'authors'}}]}
            memories = []
            if 'EvidenceK.' in text:
                memories.append({'subtype': 'hypothesis', 'statement': 'A hypothesis',
                                 'evidence': 'EvidenceK.', 'details': {'entities': [],
                                 'conditions': '', 'attribution': 'authors'}})
            return {'memories': memories}
        memories = []
        if 'EvidenceE.' in text:
            memories.append({'memory_type': 'experiment', 'subtype': 'procedure',
                             'statement': 'A procedure', 'evidence': 'EvidenceE.',
                             'details': {'objective': None, 'sample_or_product': None,
                                         'reagents': [], 'steps': [], 'conditions': [],
                                         'characterization': [], 'results': [], 'yield': None,
                                         'missing_information': []}})
        return {'memories': memories}


class WorkflowTests(unittest.TestCase):
    def test_two_types_have_page_and_character_evidence_and_repeat_deterministically(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / 'paper.pdf'
            make_pdf(pdf, ['Introduction.', 'x' * 45 + 'EvidenceK.' + 'y' * 20 + 'EvidenceE.'])
            output = root / 'out'
            client = FakeClient()
            result = run(pdf, output, client=client, chunk_size=40, chunk_overlap=5)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual((result['knowledge'], result['experiment']), (1, 1))
            snapshot = Path(result['run_dir'])
            chunks = {chunk['chunk_id']: chunk for chunk in rows(snapshot, 'chunks')}
            for kind, quote in [('knowledge', 'EvidenceK.'), ('experiment', 'EvidenceE.')]:
                memory = rows(snapshot, kind)[0]
                source = memory['source']
                chunk = chunks[source['chunk_id']]
                self.assertEqual(source['page'], 2)
                self.assertEqual(source['quote'], quote)
                self.assertGreater(chunk['start_offset'], 0)
                self.assertEqual(source['page_char_start'], chunk['start_offset'] + chunk['text'].index(quote))
                self.assertEqual(source['page_char_end'], source['page_char_start'] + len(quote))
            files = {kind: (snapshot / f'{kind}.jsonl').read_bytes()
                     for kind in ['knowledge', 'experiment', 'documents', 'chunks', 'errors']}
            call_count = len(client.calls)
            again = run(pdf, output, client=client, chunk_size=40, chunk_overlap=5)
            self.assertEqual(len(client.calls), call_count)
            self.assertEqual(again['model_calls'], 0)
            self.assertEqual(again['cached_calls'], call_count)
            for kind, expected in files.items():
                self.assertEqual((Path(again['run_dir']) / f'{kind}.jsonl').read_bytes(), expected)

    def test_only_failed_type_retries_and_successful_calls_are_cached(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / 'paper.pdf'
            make_pdf(pdf, ['EvidenceK. EvidenceE.'])
            client = FakeClient()
            client.fail_experiment_once = True
            first = run(pdf, root / 'out', client=client)
            self.assertEqual(first['status'], 'partial')
            self.assertEqual(first['errors'], 1)
            self.assertEqual(rows(Path(first['run_dir']), 'errors')[0]['stage'], 'experiment')
            self.assertEqual(first['knowledge'], 1)
            second = run(pdf, root / 'out', client=client)
            self.assertEqual(second['status'], 'complete')
            self.assertEqual((second['model_calls'], second['cached_calls']), (1, 1))
            self.assertEqual((second['knowledge'], second['experiment']), (1, 1))
            self.assertEqual([call[0] for call in client.calls], ['knowledge', 'experiment', 'experiment'])

    def test_invalid_evidence_is_rejected_and_not_cached(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / 'paper.pdf'
            make_pdf(pdf, ['EvidenceK.'])
            client = FakeClient()
            client.bad_knowledge_once = True
            first = run(pdf, root / 'out', client=client, types=('knowledge',))
            self.assertEqual(first['status'], 'partial')
            self.assertEqual(rows(Path(first['run_dir']), 'knowledge'), [])
            self.assertIn('not a verbatim substring', rows(Path(first['run_dir']), 'errors')[0]['error'])
            second = run(pdf, root / 'out', client=client, types=('knowledge',))
            self.assertEqual(second['status'], 'complete')
            self.assertEqual((second['model_calls'], second['cached_calls']), (1, 0))
            self.assertEqual(second['knowledge'], 1)

    def test_parse_only_with_corrupt_pdf_reports_partial_without_calling_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'input'
            inputs.mkdir()
            make_pdf(inputs / 'good.pdf', ['EvidenceK.'])
            (inputs / 'bad.pdf').write_bytes(b'not a PDF')
            client = FakeClient()
            result = run(inputs, root / 'out', parse_only=True, client=client)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual((result['input_pdf_count'], result['documents'], result['errors']), (2, 1, 1))
            self.assertEqual(result['model_calls'], 0)
            self.assertEqual(client.calls, [])
            snapshot = Path(result['run_dir'])
            self.assertEqual(rows(snapshot, 'errors')[0]['stage'], 'pdf')
            self.assertFalse((snapshot / 'knowledge.jsonl').exists())
            self.assertFalse((snapshot / 'experiment.jsonl').exists())
            self.assertGreater(len(rows(snapshot, 'chunks')), 0)


if __name__ == '__main__':
    unittest.main()
