
from pathlib import Path

from src.agentic_rag.ingestion import (
    clean_text,
    read_text_file,
    chunk_text,
)

file_path = Path("data/documents/travel_policy.txt")

raw_text = read_text_file(file_path)
cleaned_text = clean_text(raw_text)

print("Cleaned text:")
print(cleaned_text)

chunks = chunk_text(
    cleaned_text,
    chunk_size=100,
    overlap=20,
)

print("\n--- TEXT CHUNKS ---")

for index, chunk in enumerate(chunks):
    print(f"\nChunk {index}:")
    print(chunk)
