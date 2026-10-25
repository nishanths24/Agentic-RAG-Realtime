from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from agentic_rag.ingestion import chunk_text, clean_text, ingest_directory, read_text_file

file_path = Path("data/documents/travel_policy.txt")
raw_text = read_text_file(file_path)
cleaned_text = clean_text(raw_text)

print("Cleaned text:")
print(cleaned_text)

chunks = chunk_text(cleaned_text, chunk_size=100, overlap=20)
print("\n--- TEXT CHUNKS ---")
for index, chunk in enumerate(chunks):
    print(f"\nChunk {index}:")
    print(chunk)

_, _, report = ingest_directory(Path("data/documents"), chunk_size=1000, overlap=150)
print(
    f"\nDirectory ingestion: {len(report.processed)} documents, "
    f"{report.chunk_count} chunks, {len(report.skipped)} skipped, {len(report.failed)} failed"
)
for skip in report.skipped:
    print(f"  skipped {skip.path}: {skip.reason}")
