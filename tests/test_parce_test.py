from adv_rag.file_parser import FileParser
from pathlib import Path


def test_txt_file(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("Hello, world!")
    content = FileParser.read(file)
    assert "Hello" in content


def test_csv_file(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text("col1,col2\n1,2\n3,4")
    content = FileParser.read(csv)
    assert "col1" in content
    assert "1" in content
