
import unittest

from src.agentic_rag.ingestion import (
    clean_text,
    chunk_text,
)


class TestIngestion(unittest.TestCase):

    def test_clean_text(self):
        self.assertEqual(
            clean_text("  hello   world\n"),
            "hello world",
        )

    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])

    def test_short_text(self):
        self.assertEqual(
            chunk_text("hello", 10, 2),
            ["hello"],
        )

    def test_chunk_overlap(self):
        self.assertEqual(
            chunk_text("abcdefghij", 6, 2),
            ["abcdef", "efghij"],
        )

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", 0, 0)

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", 5, 5)


if __name__ == "__main__":
    unittest.main()
