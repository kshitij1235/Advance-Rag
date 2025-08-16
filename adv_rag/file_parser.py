from pathlib import Path
import polars as pl
from PyPDF2 import PdfReader


class FileParser:
    @staticmethod
    def read(p: Path):
        ext = p.suffix.lower()
        if ext == ".pdf":
            try:
                with open(p, "rb") as f:
                    r = PdfReader(f)
                    return "\n".join([(pg.extract_text() or "") for pg in r.pages])
            except Exception as e:
                print(f"[ERR PDF] {p}: {e}")
                return None

        if ext in {".txt", ".md"}:
            for enc in ("utf8", "utf-16", "latin1"):
                try:
                    return p.read_text(encoding=enc)
                except Exception:
                    continue
            return None

        if ext in {".xlsx", ".xls"}:
            try:
                return pl.read_excel(p).write_csv()
            except Exception as e:
                print(f"[ERR XLSX] {p}: {e}")
                return None

        if ext == ".csv":
            try:
                return pl.read_csv(p, ignore_errors=True).write_csv()
            except Exception as e:
                print(f"[ERR CSV] {p}: {e}")
                return None

        return None
