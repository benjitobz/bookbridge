"""BOOKLORE_SHELF_REQUIRE_ALIGNMENT: the Grimmory sync shelf only gets books
once they are aligned, via the daemon reconcile instead of at match time."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.web_server as web_server
from src.api.booklore_client import BookloreClient

KEYS = ("BOOKLORE_SHELF_REQUIRE_ALIGNMENT", "BOOKLORE_SHELF_NAME")


class RequireAlignmentTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in KEYS}
        os.environ["BOOKLORE_SHELF_REQUIRE_ALIGNMENT"] = "true"
        os.environ["BOOKLORE_SHELF_NAME"] = "ABS Synced"
        self._saved_db = web_server.database_service
        self._saved_globals = web_server._global_clients
        self.db = MagicMock()
        web_server.database_service = self.db
        self.client = MagicMock()
        self.client.is_configured.return_value = True
        web_server._global_clients = SimpleNamespace(booklore_client=self.client)

    def tearDown(self):
        web_server.database_service = self._saved_db
        web_server._global_clients = self._saved_globals
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _book(abs_id, source_id, source="Grimmory"):
    return SimpleNamespace(abs_id=abs_id, abs_title=abs_id, ebook_source=source, ebook_source_id=source_id)


class TestReconcile(RequireAlignmentTestCase):
    def test_off_does_nothing(self):
        os.environ["BOOKLORE_SHELF_REQUIRE_ALIGNMENT"] = "false"
        web_server._reconcile_aligned_shelf()
        self.db.get_books_by_status.assert_not_called()

    def test_adds_only_aligned_grimmory_books_that_are_missing(self):
        self.db.get_books_by_status.return_value = [
            _book("a1", "11"), _book("a2", "22"), _book("a3", "33"), _book("a4", "44", source="ABS"),
        ]
        self.db.has_alignment.side_effect = lambda abs_id: abs_id in ("a1", "a2")
        self.client.list_books_on_shelf.return_value = [{"id": 22}]
        self.client.add_book_id_to_shelf.return_value = True

        added = web_server._reconcile_aligned_shelf()

        self.client.list_books_on_shelf.assert_called_once_with("ABS Synced")
        self.client.add_book_id_to_shelf.assert_called_once_with("11", "ABS Synced")
        self.assertEqual(1, added)

    def test_nothing_aligned_never_touches_grimmory(self):
        self.db.get_books_by_status.return_value = [_book("a1", "11")]
        self.db.has_alignment.return_value = False
        web_server._reconcile_aligned_shelf()
        self.client.list_books_on_shelf.assert_not_called()

    def test_unconfigured_client_is_skipped(self):
        self.client.is_configured.return_value = False
        web_server._reconcile_aligned_shelf()
        self.db.get_books_by_status.assert_not_called()

    def test_errors_are_swallowed(self):
        self.db.get_books_by_status.side_effect = RuntimeError("boom")
        web_server._reconcile_aligned_shelf()


class TestMatchTimeShelving(RequireAlignmentTestCase):
    def test_grimmory_add_is_deferred_when_required(self):
        with patch.object(web_server, "uc", return_value=SimpleNamespace(booklore_client=self.client)), \
                patch.object(web_server, "_is_abs_hosted_ebook_filename", return_value=False):
            web_server._shelve_matched_ebook("book.epub", "grimmory", "11")
        self.client.add_to_shelf.assert_not_called()
        self.client.add_book_id_to_shelf.assert_not_called()

    def test_grimmory_add_still_happens_when_off(self):
        os.environ["BOOKLORE_SHELF_REQUIRE_ALIGNMENT"] = "false"
        self.client.add_to_shelf.return_value = True
        with patch.object(web_server, "uc", return_value=SimpleNamespace(booklore_client=self.client)), \
                patch.object(web_server, "_is_abs_hosted_ebook_filename", return_value=False):
            web_server._shelve_matched_ebook("book.epub", "grimmory", "11")
        self.client.add_to_shelf.assert_called_once_with("book.epub", "ABS Synced")


class TestAddBookIdToShelf(unittest.TestCase):
    def _client(self, shelf_id, response):
        client = BookloreClient.__new__(BookloreClient)
        client._creds = None
        client._get_or_create_shelf_id = MagicMock(return_value=shelf_id)
        client._make_request = MagicMock(return_value=response)
        return client

    def test_assigns_by_id(self):
        client = self._client(18, MagicMock(status_code=200))
        self.assertTrue(client.add_book_id_to_shelf("45", "ABS Synced"))
        client._make_request.assert_called_once_with("POST", "/api/v1/books/shelves", {
            "bookIds": [45], "shelvesToAssign": [18], "shelvesToUnassign": []})

    def test_missing_shelf_or_failure_is_false(self):
        self.assertFalse(self._client(None, MagicMock(status_code=200)).add_book_id_to_shelf("45", "ABS Synced"))
        self.assertFalse(self._client(18, MagicMock(status_code=401)).add_book_id_to_shelf("45", "ABS Synced"))
        self.assertFalse(self._client(18, MagicMock(status_code=200)).add_book_id_to_shelf("", "ABS Synced"))


if __name__ == "__main__":
    unittest.main()
