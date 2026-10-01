"""Regression tests for #454: ABS-acquired cache names skip the library walks.

``LibraryService.acquire_ebook`` names the ebook it downloads for an ABS item
``<item_id>_direct.<ext>`` (or ``_cwa`` / ``_abs_search``) and writes it into the
epub cache. ``EbookParser.resolve_book_path`` only checked the cache after two full
recursive walks of every library dir, so on a CIFS-mounted library each BridgeSync
download spent ~70s walking before its first byte and the plugin's 30s stall timeout
fired (``Request interrupted: wantread``).
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.db.database_service import DatabaseService
from src.db.models import Book
from src.services.koreader_device_sync_service import KOReaderDeviceSyncService
from src.utils.ebook_utils import EbookParser, is_managed_cache_filename

ABS_ITEM_ID = "3f0c7d9e-5b8a-4e21-9c3d-1a2b3c4d5e6f"


class _LibraryWalkSpy:
    """Record every Path.glob / Path.rglob call made under the library root."""

    def __init__(self, library_root: Path):
        self.library_root = str(library_root.resolve())
        self.walks: list[tuple[str, str]] = []
        self._real_glob = Path.glob
        self._real_rglob = Path.rglob

    def _record(self, kind: str, path: Path, pattern: str) -> None:
        if str(Path(path).resolve()).startswith(self.library_root):
            self.walks.append((kind, pattern))

    def __enter__(self):
        spy = self

        def glob(path_self, pattern, *args, **kwargs):
            spy._record("glob", path_self, pattern)
            return spy._real_glob(path_self, pattern, *args, **kwargs)

        def rglob(path_self, pattern, *args, **kwargs):
            spy._record("rglob", path_self, pattern)
            return spy._real_rglob(path_self, pattern, *args, **kwargs)

        self._patches = [
            patch.object(Path, "glob", glob),
            patch.object(Path, "rglob", rglob),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


def _make_dirs(tmp: Path) -> tuple[Path, Path]:
    books = tmp / "books"
    cache = tmp / "epub_cache"
    books.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    # A small library tree so a walk has something to traverse.
    for author in ("Author A", "Author B"):
        series = books / author / "Series"
        series.mkdir(parents=True, exist_ok=True)
        (series / f"{author} - Book.epub").write_bytes(b"library epub")
    return books, cache


class TestDeviceSyncDownloadOfDirectBook(unittest.TestCase):
    """The reported path: BridgeSync download of an ABS-direct book."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bb454_"))
        self.books_dir, self.cache_dir = _make_dirs(self.tmp)
        self.db = DatabaseService(str(self.tmp / "test.db"))
        self.parser = EbookParser(books_dir=str(self.books_dir), epub_cache_dir=str(self.cache_dir))
        self.service = KOReaderDeviceSyncService(
            database_service=self.db,
            ebook_parser=self.parser,
            abs_client=MagicMock(),
            booklore_client=MagicMock(),
            cwa_client=MagicMock(),
            kavita_client=MagicMock(),
            epub_cache_dir=self.cache_dir,
            bookorbit_client=MagicMock(),
        )

    def tearDown(self):
        self.db.db_manager.engine.dispose()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_download_serves_cached_direct_ebook_without_walking_library(self):
        filename = f"{ABS_ITEM_ID}_direct.epub"
        cached = self.cache_dir / filename
        cached.write_bytes(b"abs direct epub bytes" * 100)
        self.db.save_book(Book(
            abs_id=ABS_ITEM_ID,
            abs_title="Direct Book",
            ebook_filename=filename,
            original_ebook_filename=filename,
            status="active",
        ))

        with _LibraryWalkSpy(self.books_dir) as spy:
            resolved = self.service.resolve_download(ABS_ITEM_ID)

        self.assertIsNotNone(resolved)
        self.assertEqual(Path(resolved["path"]).resolve(), cached.resolve())
        self.assertEqual(resolved["content_hash"], self.parser.get_kosync_id(cached))
        self.assertEqual(
            spy.walks, [],
            "a cached <abs_id>_direct.epub must not trigger a library walk "
            "(two full walks exceeded the BridgeSync stall timeout on CIFS)",
        )

    def test_manifest_build_does_not_walk_library_for_direct_books(self):
        for idx in range(3):
            abs_id = f"{ABS_ITEM_ID[:-1]}{idx}"
            filename = f"{abs_id}_direct.epub"
            (self.cache_dir / filename).write_bytes(b"epub" * 400 + bytes([idx]))
            self.db.save_book(Book(
                abs_id=abs_id,
                abs_title=f"Direct Book {idx}",
                ebook_filename=filename,
                original_ebook_filename=filename,
                status="active",
            ))

        with _LibraryWalkSpy(self.books_dir) as spy:
            manifest = self.service.build_manifest()

        self.assertEqual(len(manifest["books"]), 3)
        self.assertEqual(spy.walks, [])


class TestResolveBookPathGeneratedCacheNames(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bb454_"))
        self.books_dir, self.cache_dir = _make_dirs(self.tmp)
        self.parser = EbookParser(books_dir=str(self.books_dir), epub_cache_dir=str(self.cache_dir))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_cache_hit_without_walk(self, filename: str) -> None:
        cached = self.cache_dir / filename
        cached.write_bytes(b"cached epub")
        with _LibraryWalkSpy(self.books_dir) as spy:
            result = self.parser.resolve_book_path(filename)
        self.assertEqual(Path(result).resolve(), cached.resolve())
        self.assertEqual(spy.walks, [], filename)
        self.assertIn(filename, self.parser._path_cache)

    def test_every_acquisition_suffix_resolves_from_cache_first(self):
        for suffix in ("direct", "cwa", "abs_search"):
            with self.subTest(suffix=suffix):
                self._assert_cache_hit_without_walk(f"{ABS_ITEM_ID}_{suffix}.epub")

    def test_non_uuid_item_ids_and_other_extensions(self):
        # Pre-2.3 ABS item ids ("li_...") and non-EPUB formats use the same scheme.
        for filename in ("li_8gch9ve09orgn4fdz8_direct.epub", f"{ABS_ITEM_ID}_direct.pdf"):
            with self.subTest(filename=filename):
                self._assert_cache_hit_without_walk(filename)

    def test_generated_name_missing_from_cache_still_searches_library(self):
        filename = f"{ABS_ITEM_ID}_direct.epub"
        lib_file = self.books_dir / "Author A" / filename
        lib_file.write_bytes(b"copied into the library by hand")
        self.assertEqual(self.parser.resolve_book_path(filename), lib_file)

    def test_generated_name_missing_everywhere_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.parser.resolve_book_path(f"{ABS_ITEM_ID}_direct.epub")

    def test_ordinary_name_keeps_library_precedence_over_cache(self):
        # Control: only generated names check the cache first. A library file and
        # a same-named cached copy must still resolve to the library file.
        filename = "Author A - Book.epub"
        (self.cache_dir / filename).write_bytes(b"older cached copy")
        result = self.parser.resolve_book_path(filename)
        self.assertEqual(result, self.books_dir / "Author A" / "Series" / filename)


class TestIsManagedCacheFilename(unittest.TestCase):
    def test_managed_names(self):
        for name in (
            "bookfusion_abc123.epub",
            "storyteller_uuid-42.epub",
            f"{ABS_ITEM_ID}_direct.epub",
            f"{ABS_ITEM_ID}_cwa.epub",
            f"{ABS_ITEM_ID}_abs_search.epub",
            "li_8gch9ve09orgn4fdz8_direct.pdf",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_managed_cache_filename(name))

    def test_ordinary_names(self):
        for name in (
            "The Direct Approach.epub",
            "the_direct_approach.epub",
            "_direct.epub",
            "book_direct.epub.bak.",
            "direct.epub",
            "Author - Title_cwa",
            "",
            None,
        ):
            with self.subTest(name=name):
                self.assertFalse(is_managed_cache_filename(name))


if __name__ == "__main__":
    unittest.main()
