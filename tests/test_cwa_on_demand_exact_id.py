"""Regression tests for GitHub issue #448 — CWA on-demand download grabbing the
wrong book.

`get_kosync_id_for_ebook`'s CWA on-demand block searches CWA by the numeric
Calibre id as free text, then treated a single search result as a match even
when that result's id did not match the requested one. Reporter: a book
titled "... (9781101619001)" was the only OPDS search hit for CWA ids 9, 10,
11, 16, and 19 (the digits happened to match the title's embedded ISBN), so
the bridge downloaded and cached that wrong EPUB under each of those ids.
`cwa_client.get_book_by_id` had the identical `len(results) == 1` fallback in
its own per-endpoint loop.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.web_server as web_server
from src.api.cwa_client import CWAClient


def _opds_feed(entry_id, title="Some Book"):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <title>{title}</title>
            <author><name>Author</name></author>
            <link rel="http://opds-spec.org/acquisition" type="application/epub+zip" href="/opds/download/{entry_id}/epub/" />
        </entry>
    </feed>
    """


class TestCwaGetBookByIdExactMatch(unittest.TestCase):
    """cwa_client.get_book_by_id must reject a single non-matching OPDS hit (#448)."""

    def setUp(self):
        self.env_patcher = patch.dict('os.environ', {
            'CWA_ENABLED': 'true',
            'CWA_SERVER': 'http://cwa:8083',
            'CWA_USERNAME': 'user',
            'CWA_PASSWORD': 'pass',
        })
        self.env_patcher.start()
        self.client = CWAClient()

    def tearDown(self):
        self.env_patcher.stop()

    @patch('requests.Session.get')
    def test_single_non_matching_result_falls_through_to_fallback(self, mock_get):
        # Every endpoint tried returns the same feed: one entry, id 13 — never
        # the requested id 9 (title carries the requested digits as an ISBN).
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = _opds_feed(
            "13", title="Myth of the Perfect Girl : ... (9781101619001)"
        )

        result = self.client.get_book_by_id("9")

        self.assertEqual(result["source"], "CWA_Fallback")
        self.assertTrue(result["download_url"].endswith("/opds/download/9/epub/"))

    @patch('requests.Session.get')
    def test_exact_id_match_is_returned(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = _opds_feed("9", title="Correct Book")

        result = self.client.get_book_by_id("9")

        self.assertEqual(result["id"], "9")
        self.assertEqual(result["title"], "Correct Book")


class TestCwaOnDemandExactIdMatch(unittest.TestCase):
    """get_kosync_id_for_ebook's CWA on-demand block must not accept an
    ambiguous single search result (#448)."""

    _WRONG_HIT = {
        "id": "13",
        "title": "Myth of the Perfect Girl : ... (9781101619001)",
        "download_url": "http://cwa/opds/download/13/epub/",
    }

    @staticmethod
    def _clients(cwa_client):
        bookorbit = MagicMock()
        bookorbit.is_configured.return_value = False
        booklore = MagicMock()
        booklore.is_configured.return_value = False
        return SimpleNamespace(
            bookorbit_client=bookorbit, booklore_client=booklore, cwa_client=cwa_client
        )

    @staticmethod
    def _container(tmp_cache, kosync_id="hashed-id"):
        parser = MagicMock()
        parser.get_kosync_id.return_value = kosync_id
        container = MagicMock()
        container.epub_cache_dir.return_value = Path(tmp_cache)
        container.ebook_parser.return_value = parser
        return container

    def test_reporter_ids_do_not_download_the_title_hit(self):
        for cwa_id in ("9", "10", "11", "16", "19"):
            with self.subTest(cwa_id=cwa_id):
                cwa_client = MagicMock()
                cwa_client.is_configured.return_value = True
                cwa_client.search_ebooks.return_value = [dict(self._WRONG_HIT)]
                correct_url = f"http://cwa/opds/download/{cwa_id}/epub/"
                cwa_client.get_book_by_id.return_value = {
                    "id": cwa_id,
                    "title": f"Real Book {cwa_id}",
                    "download_url": correct_url,
                    "source": "CWA_Fallback",
                }
                cwa_client.download_ebook.return_value = True

                with tempfile.TemporaryDirectory() as tmp:
                    container = self._container(tmp)
                    with patch.object(web_server, "uc", return_value=self._clients(cwa_client)), \
                         patch.object(web_server, "container", container), \
                         patch.object(web_server, "find_ebook_file", return_value=None), \
                         patch.object(web_server, "EBOOK_DIR", Path(tmp) / "no-books", create=True):
                        result = web_server.get_kosync_id_for_ebook(f"cwa_{cwa_id}.epub")

                self.assertEqual(result, "hashed-id")
                cwa_client.get_book_by_id.assert_called_once_with(cwa_id)
                cwa_client.download_ebook.assert_called_once_with(correct_url, ANY)

    def test_exact_id_search_hit_skips_get_book_by_id(self):
        cwa_client = MagicMock()
        cwa_client.is_configured.return_value = True
        cwa_client.search_ebooks.return_value = [{
            "id": "20",
            "title": "The Right Book",
            "download_url": "http://cwa/opds/download/20/epub/",
        }]
        cwa_client.download_ebook.return_value = True

        with tempfile.TemporaryDirectory() as tmp:
            container = self._container(tmp)
            with patch.object(web_server, "uc", return_value=self._clients(cwa_client)), \
                 patch.object(web_server, "container", container), \
                 patch.object(web_server, "find_ebook_file", return_value=None), \
                 patch.object(web_server, "EBOOK_DIR", Path(tmp) / "no-books", create=True):
                result = web_server.get_kosync_id_for_ebook("cwa_20.epub")

        self.assertEqual(result, "hashed-id")
        cwa_client.get_book_by_id.assert_not_called()
        cwa_client.download_ebook.assert_called_once_with(
            "http://cwa/opds/download/20/epub/", ANY
        )

    def test_no_match_logs_info_before_id_lookup(self):
        cwa_client = MagicMock()
        cwa_client.is_configured.return_value = True
        cwa_client.search_ebooks.return_value = [dict(self._WRONG_HIT)]
        cwa_client.get_book_by_id.return_value = None

        with tempfile.TemporaryDirectory() as tmp:
            container = self._container(tmp)
            with patch.object(web_server, "uc", return_value=self._clients(cwa_client)), \
                 patch.object(web_server, "container", container), \
                 patch.object(web_server, "find_ebook_file", return_value=None), \
                 patch.object(web_server, "EBOOK_DIR", Path(tmp) / "no-books", create=True), \
                 self.assertLogs("src.web_server", level="INFO") as captured:
                web_server.get_kosync_id_for_ebook("cwa_9.epub")

        expected = (
            "🔍 CWA search for ID '9' returned 1 result(s), none with that ID "
            "— looking it up by ID"
        )
        self.assertTrue(any(expected in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()
