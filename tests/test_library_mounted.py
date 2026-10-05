import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.services.library_service import LibraryService


class TestMountedLibrary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.books = self.root / 'books'
        self.books.mkdir()
        self.cache = self.root / 'cache'
        self.client = MagicMock()
        self.client.download_book.return_value = b'x' * 2048
        self.service = LibraryService(MagicMock(), self.client, None, None, self.cache)
        self.book = SimpleNamespace(ebook_source='Grimmory', ebook_source_id='42',
                                    ebook_filename='comic [1].cbz', original_ebook_filename=None)
        env = patch.dict(os.environ, BOOKS_DIR=str(self.books), EXTRA_EBOOK_DIRS='')
        env.start()
        self.addCleanup(env.stop)

    def test_mounted_file_precedes_existing_cache(self):
        nested = self.books / 'author'
        nested.mkdir()
        local = nested / self.book.ebook_filename
        local.write_bytes(b'x' * 2048)
        (self.cache / self.book.ebook_filename).write_bytes(b'y' * 2048)
        self.assertEqual(self.service.acquire_ebook({}, self.book), str(local))
        self.client.download_book.assert_not_called()

    def test_missing_local_file_uses_existing_cache(self):
        cached = self.cache / self.book.ebook_filename
        cached.write_bytes(b'x' * 2048)
        self.assertEqual(self.service.acquire_ebook({}, self.book), str(cached))
        self.client.download_book.assert_not_called()

    def test_missing_local_and_cache_downloads(self):
        result = self.service.acquire_ebook({}, self.book)
        self.assertEqual(result, str(self.cache / self.book.ebook_filename))
        self.client.download_book.assert_called_once_with('42')

    def test_renamed_file_in_extra_directory(self):
        self.book.original_ebook_filename = 'old.cbz'
        extra = self.root / 'extra'
        extra.mkdir()
        local = extra / self.book.ebook_filename
        local.write_bytes(b'x' * 2048)
        with patch.dict(os.environ, EXTRA_EBOOK_DIRS=f'{self.root / "missing"},\n {extra}'):
            self.assertEqual(self.service.acquire_ebook({}, self.book), str(local))
        self.client.download_book.assert_not_called()

    def test_original_filename_still_resolves(self):
        self.book.original_ebook_filename = 'old.cbz'
        local = self.books / self.book.original_ebook_filename
        local.write_bytes(b'x' * 2048)
        self.assertEqual(self.service.acquire_ebook({}, self.book), str(local))
        self.client.download_book.assert_not_called()

    def test_unsafe_name_does_not_escape_library(self):
        self.book.ebook_filename = '../secret.cbz'
        (self.root / 'secret.cbz').write_bytes(b'x' * 2048)
        self.assertIsNone(self.service.acquire_ebook({}, self.book))
        self.client.download_book.assert_not_called()

    def test_empty_mounted_file_is_not_used(self):
        (self.books / self.book.ebook_filename).touch()
        result = self.service.acquire_ebook({}, self.book)
        self.assertEqual(result, str(self.cache / self.book.ebook_filename))
        self.client.download_book.assert_called_once_with('42')

    def test_unavailable_mount_uses_existing_cache(self):
        cached = self.cache / self.book.ebook_filename
        cached.write_bytes(b'x' * 2048)
        with patch.object(Path, 'glob', side_effect=OSError('NAS unavailable')):
            self.assertEqual(self.service.acquire_ebook({}, self.book), str(cached))
        self.client.download_book.assert_not_called()
