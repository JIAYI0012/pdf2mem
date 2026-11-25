from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from pdf_to_agent_memory.pdf import PDFExtractionError, extract_pdf


def make_pdf(path: Path, page_texts: list[str], *, password: str | None = None) -> None:
    writer = PdfWriter()
    writer.add_metadata({"/Title": "Test article"})
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {NameObject("/Type"): NameObject("/Font"),
             NameObject("/Subtype"): NameObject("/Type1"),
             NameObject("/BaseFont"): NameObject("/Helvetica")}
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    with path.open("wb") as output:
        writer.write(output)


def check_page_provenance_overlap_stability_and_metadata(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path, ["First page " + "word " * 30, "Second page", ""])
    doc = extract_pdf(path, chunk_size=50, chunk_overlap=10)

    assert doc.metadata["Title"] == "Test article"
    assert doc.metadata["page_count"] == 3
    assert doc.empty_pages == [3]
    first_page = [chunk for chunk in doc.chunks if chunk.page_start == 1]
    assert len(first_page) > 1
    for chunk in doc.chunks:
        assert chunk.page_start == chunk.page_end
        assert chunk.text == doc.pages[chunk.page_start - 1].text[chunk.start_offset:chunk.end_offset]
        assert len(chunk.text) <= 50
    for left, right in zip(first_page, first_page[1:]):
        assert left.end_offset - right.start_offset == 10
        assert left.text[-10:] == right.text[:10]

    renamed = tmp_path / "copy.pdf"
    renamed.write_bytes(path.read_bytes())
    copied = extract_pdf(renamed, chunk_size=50, chunk_overlap=10)
    assert copied.document_id == doc.document_id
    assert [chunk.chunk_id for chunk in copied.chunks] == [chunk.chunk_id for chunk in doc.chunks]
    assert len({chunk.chunk_id for chunk in doc.chunks}) == len(doc.chunks)


def check_errors_for_invalid_or_textless_files(tmp_path, case):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a PDF")
    with case.assertRaisesRegex(PDFExtractionError, "Cannot parse"):
        extract_pdf(broken)
    with case.assertRaisesRegex(PDFExtractionError, "Cannot read"):
        extract_pdf(tmp_path / "missing.pdf")
    blank = tmp_path / "blank.pdf"
    make_pdf(blank, [""])
    with case.assertRaisesRegex(PDFExtractionError, "No extractable text"):
        extract_pdf(blank)
    locked = tmp_path / "locked.pdf"
    make_pdf(locked, ["locked"], password="secret")
    with case.assertRaisesRegex(PDFExtractionError, "password-protected"):
        extract_pdf(locked)


class PDFTests(unittest.TestCase):
    def test_page_provenance_overlap_stability_and_metadata(self):
        with TemporaryDirectory() as directory:
            check_page_provenance_overlap_stability_and_metadata(Path(directory))

    def test_errors_for_invalid_or_textless_files(self):
        with TemporaryDirectory() as directory:
            check_errors_for_invalid_or_textless_files(Path(directory), self)

    def test_rejects_invalid_chunk_settings(self):
        for size, overlap in [(0, 0), (10, -1), (10, 10), (10, 11)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                extract_pdf("unused.pdf", chunk_size=size, chunk_overlap=overlap)

    def test_page_extraction_failure_is_not_silently_skipped(self):
        reader = MagicMock()
        reader.is_encrypted = False
        reader.metadata = {}
        good_page, bad_page = MagicMock(), MagicMock()
        good_page.extract_text.return_value = "some text"
        bad_page.extract_text.side_effect = ValueError("bad text layer")
        reader.pages = [good_page, bad_page]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            path.write_bytes(b"mock content")
            with patch("pdf_to_agent_memory.pdf.PdfReader", return_value=reader):
                with self.assertRaisesRegex(PDFExtractionError, "page 2"):
                    extract_pdf(path)
