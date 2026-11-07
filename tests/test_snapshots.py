import hashlib
from pathlib import Path
import tempfile
import unittest

from src.contracts import CorpusResult, UDLException
from src.snapshots import snapshot_corpus


class SnapshotTests(unittest.TestCase):
    def test_updates_cannot_rewrite_prior_source_or_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data/raw/book.pdf"
            crop = root / "data/assets/equation.png"
            source.parent.mkdir(parents=True)
            crop.parent.mkdir(parents=True)
            source.write_bytes(b"original source")
            crop.write_bytes(b"original crop")
            corpus = CorpusResult(sources=[{"path": "data/raw/book.pdf", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
                                  chunks=[{"metadata": {"asset_path": "data/assets/equation.png"}}])
            saved = snapshot_corpus(corpus, root)
            source.write_bytes(b"new source")
            crop.write_bytes(b"new crop")
            self.assertEqual((root / saved.sources[0]["path"]).read_bytes(), b"original source")
            self.assertEqual((root / saved.chunks[0]["metadata"]["asset_path"]).read_bytes(), b"original crop")
            self.assertEqual(corpus.sources[0]["path"], "data/raw/book.pdf")
            with self.assertRaises(UDLException):
                snapshot_corpus(corpus, root)

    def test_source_symlink_cannot_escape_data_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private = root / "private.txt"
            private.write_bytes(b"private")
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            (raw / "book.pdf").symlink_to(private)
            corpus = CorpusResult(sources=[{"path": "data/raw/book.pdf", "sha256": hashlib.sha256(b"private").hexdigest()}])
            with self.assertRaises(UDLException):
                snapshot_corpus(corpus, root)


if __name__ == "__main__":
    unittest.main()
