from pathlib import Path
import re
import fitz  
import polars as pl

class FileParser:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize PDF-extracted text:
        - Remove page numbers, headers, footers, activities
        - Merge broken lines into paragraphs
        - Normalize whitespace
        - Remove duplicate consecutive lines
        """
        # Remove lines with only numbers (page numbers)
        text = "\n".join([line for line in text.splitlines() if not re.fullmatch(r"\s*\d+\s*", line)])
        
        # Remove common headers/footers
        text = re.sub(r'(Activity\s*:.*|Figure\s*\d+.*|Table\s*\d+.*)', '', text, flags=re.IGNORECASE)
        
        # Merge broken lines into paragraphs
        lines = text.splitlines()
        merged = []
        buffer = ""
        for line in lines:
            line = line.strip()
            if not line:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                continue
            if buffer and not re.search(r"[.!?]$", buffer):
                buffer += " " + line
            else:
                if buffer:
                    merged.append(buffer)
                buffer = line
        if buffer:
            merged.append(buffer)

        # Normalize whitespace
        merged = [re.sub(r"\s+", " ", l) for l in merged]

        # Remove consecutive duplicate lines
        final_lines = []
        prev = None
        for line in merged:
            if line != prev:
                final_lines.append(line)
            prev = line

        return "\n\n".join(final_lines)

    @staticmethod
    def chunk_text(text: str, max_tokens: int = 500) -> list:
        """
        Split text into coherent chunks for RAG indexing.
        Default max_tokens ~500 words per chunk.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_tokens):
            chunk = " ".join(words[i:i + max_tokens])
            chunks.append(chunk)
        return chunks

    @staticmethod
    def read(p: Path, chunked: bool = False):
        ext = p.suffix.lower()
        text = None

        if ext == ".pdf":
            try:
                doc = fitz.open(p)
                raw_text = "\n".join([page.get_text() for page in doc])
                text = FileParser.clean_text(raw_text)
            except Exception as e:
                print(f"[ERR PDF] {p}: {e}")
                return None

        elif ext in {".txt", ".md"}:
            for enc in ("utf8", "utf-16", "latin1"):
                try:
                    text = p.read_text(encoding=enc)
                    break
                except Exception:
                    continue

        elif ext in {".xlsx", ".xls"}:
            try:
                text = pl.read_excel(p).write_csv()
            except Exception as e:
                print(f"[ERR XLSX] {p}: {e}")
                return None

        elif ext == ".csv":
            try:
                text = pl.read_csv(p, ignore_errors=True).write_csv()
            except Exception as e:
                print(f"[ERR CSV] {p}: {e}")
                return None

        if text and chunked:
            return FileParser.chunk_text(text)
        return text

