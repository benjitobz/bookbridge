import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT

from src.services.alignment_service import AlignmentService
from src.utils import ebook_dom_map
from src.utils.ebook_dom_map import build_dom_anchor_map, reconstruct_text
from src.utils.ebook_utils import INLINE_TEXT_JOINER, EbookParser


class TestEbookInlineTextExtraction(unittest.TestCase):
    def setUp(self):
        self.parser = EbookParser(books_dir=".")

    def _extract(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        return self.parser._extract_text_from_soup(soup)

    def test_bionic_inline_markup_joins_word_fragments(self):
        html_content = "<html><body><h1><b>T</b>he <b>Dung</b>eon</h1><p>Next paragraph.</p></body></html>"
        soup = BeautifulSoup(html_content, "html.parser")
        extracted = self._extract(html_content)

        self.assertEqual(extracted.replace(EbookParser.INLINE_TEXT_JOINER, ""), "The Dungeon Next paragraph.")
        self.assertGreater(extracted.count(EbookParser.INLINE_TEXT_JOINER), 0)
        self.assertEqual(
            len(extracted),
            len(soup.get_text(separator=" ", strip=True)),
        )

    def test_normal_bold_sections_keep_literal_word_boundaries(self):
        html_content = (
            "<html><body><p>This is <b>bold text</b> and "
            "<strong>more bold text</strong>.</p><p>Next paragraph.</p></body></html>"
        )

        extracted = self._extract(html_content)

        self.assertEqual(
            extracted.replace(EbookParser.INLINE_TEXT_JOINER, ""),
            "This is bold text and more bold text. Next paragraph.",
        )

    def test_extract_text_and_map_uses_inline_aware_text(self):
        content = b"<html><body><h1><b>T</b>he <b>Dung</b>eon.</h1></body></html>"
        item = SimpleNamespace(
            get_type=lambda: ITEM_DOCUMENT,
            get_content=lambda: content,
            get_name=lambda: "chapter.xhtml",
        )
        book = SimpleNamespace(
            spine=[("chapter", None)],
            get_item_with_id=lambda _item_id: item,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = Path(temp_dir) / "book.epub"
            epub_path.write_bytes(b"fixture")
            self.parser._build_href_resolver = lambda _path: lambda name: name
            with patch("src.utils.ebook_utils.epub.read_epub", return_value=book):
                extracted, spine_map = self.parser.extract_text_and_map(epub_path)

        self.assertEqual(extracted.replace(EbookParser.INLINE_TEXT_JOINER, ""), "The Dungeon.")
        self.assertEqual(spine_map[0]["char_len"], len(extracted))

    def test_non_content_script_and_style_text_is_ignored(self):
        html_content = (
            "<html><head><style>.hidden { color: red; }</style>"
            "<script>alert('ignored')</script></head><body><p>Text.</p></body></html>"
        )

        extracted = self._extract(html_content)

        self.assertEqual(extracted, "Text.")

    def test_block_boundaries_still_create_spaces(self):
        html_content = "<html><body><p>First.</p><p>Second.</p></body></html>"

        extracted = self._extract(html_content)

        self.assertEqual(extracted, "First. Second.")

    def test_inline_text_joiner_is_defined_at_module_level(self):
        """Regression for the merged PR's ``INLINE_TEXT_JOINER =
        INLINE_TEXT_JOINER`` class-body bug: that right-hand side referred to
        a module-level name that did not exist anywhere, so merely importing
        ``ebook_utils`` (and therefore ``alignment_service``, which imports
        ``INLINE_TEXT_JOINER`` from it) raised ``NameError`` -- this whole
        test file failed to collect, not just this one assertion.
        """
        self.assertEqual(len(INLINE_TEXT_JOINER), 1)
        self.assertEqual(EbookParser.INLINE_TEXT_JOINER, INLINE_TEXT_JOINER)
        self.assertIs(EbookParser.INLINE_TEXT_JOINER, ebook_dom_map.INLINE_TEXT_JOINER)

    def test_dom_anchor_map_matches_a_book_with_inline_joins(self):
        """``build_dom_anchor_map`` must not raise "DOM anchor mismatch" for a
        book with a bionic-reading inline join: it re-parses the same spine
        content independently and compares its own rebuilt text against
        ``extract_text_and_map``'s combined text, so both must make the exact
        same inline-join decision (finding 2)."""
        content = b"<html><body><h1><b>T</b>he <b>Dung</b>eon</h1><p>Next paragraph.</p></body></html>"
        item = SimpleNamespace(
            get_type=lambda: ITEM_DOCUMENT,
            get_content=lambda: content,
            get_name=lambda: "chapter.xhtml",
        )
        book = SimpleNamespace(
            spine=[("chapter", None)],
            get_item_with_id=lambda _item_id: item,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = Path(temp_dir) / "book.epub"
            epub_path.write_bytes(b"fixture")
            self.parser._build_href_resolver = lambda _path: lambda name: name
            with patch("src.utils.ebook_utils.epub.read_epub", return_value=book):
                text, _spine_map = self.parser.extract_text_and_map(epub_path)
                self.assertIn(INLINE_TEXT_JOINER, text)

                dom_map = build_dom_anchor_map(self.parser, epub_path)  # must not raise

        self.assertEqual(reconstruct_text(dom_map), text)

    def test_ruby_annotation_text_is_excluded_like_get_text(self):
        """The extractor must yield exactly the same string set bs4's own
        ``get_text()`` does. bs4 gives ``<rt>``/``<rp>`` ruby-annotation text
        its own ``NavigableString`` subclasses specifically so ``get_text()``
        skips them; a walker that accepts any ``isinstance(node,
        NavigableString)`` (the merged PR's approach) would surface them
        instead and shift every later offset (finding 3)."""
        html_content = "<p><ruby>K<rp>(</rp><rt>kan</rt><rp>)</rp></ruby>J</p>"
        soup = BeautifulSoup(html_content, "html.parser")
        extracted = self._extract(html_content)

        self.assertEqual(extracted, soup.get_text(separator=" ", strip=True))
        self.assertEqual(extracted, "K J")
        self.assertNotIn(INLINE_TEXT_JOINER, extracted)

    def test_void_elements_between_runs_do_not_join(self):
        """``<br/>`` and ``<img/>`` produce no text node of their own, but still
        separate words: verse and addresses routinely use ``line<br/>line`` with
        no source whitespace. ``<wbr>`` (a break opportunity, no space) and pure
        styling tags still join."""
        cases = {
            '<p>first line<br/>second line</p>': "first line second line",
            '<p>end<img src="x.png"/>start</p>': "end start",
            '<p>super<wbr/>cali</p>': f"super{INLINE_TEXT_JOINER}cali",
            '<p><b>Th</b>e</p>': f"Th{INLINE_TEXT_JOINER}e",
        }
        for html_content, expected in cases.items():
            with self.subTest(html=html_content):
                self.assertEqual(self._extract(html_content), expected)

    def test_adjacent_sibling_spans_do_not_join(self):
        """Two sibling ``<span>``s with no literal whitespace between them
        (a common verse-line shape) are separate words and must not fuse
        (finding 4): each is its own nearest-non-styling ancestor, so the
        join is refused and a real space is used, exactly like before this
        PR."""
        html_content = (
            '<p><span class="line">first line</span>'
            '<span class="line">second line</span></p>'
        )
        extracted = self._extract(html_content)

        self.assertNotIn(INLINE_TEXT_JOINER, extracted)
        self.assertEqual(extracted, "first line second line")

    def test_footnote_reference_does_not_join_to_preceding_word(self):
        """A footnote reference (``<a>`` wrapping ``<sup>``) directly after a
        word with no literal whitespace must not fuse into it (finding 4):
        neither ``<a>`` nor ``<sup>`` is a join-safe styling tag, so "word"
        and "1" never share a nearest-non-styling ancestor."""
        html_content = '<p>word<a href="#fn1"><sup>1</sup></a> more text.</p>'
        extracted = self._extract(html_content)

        self.assertNotIn(INLINE_TEXT_JOINER, extracted)
        self.assertEqual(extracted, "word 1 more text.")

    def test_deeply_nested_inline_markup_does_not_recurse(self):
        """Thousands of unclosed/nested inline tags must not blow the
        recursion limit (finding 7): a recursive node walker (the merged
        PR's ``walk()``) hits ``RecursionError`` here, which the caller's
        broad ``except Exception`` turns into "no text for the whole book".
        Enumerating via ``content_string_nodes`` (bs4's own iterative
        ``.descendants``) has no such limit."""
        depth = 5000  # well past Python's default recursion limit (1000)
        html_content = "<p>" + "<font>" * depth + "word" + "</font>" * depth + "</p>"

        extracted = self._extract(html_content)

        self.assertEqual(extracted, "word")

    def test_find_text_location_prefers_exact_joiner_aware_match_over_ambiguous_normalized_match(self):
        """Round-trip regression (finding 5): ``get_text_at_percentage`` (and
        similar) strip the joiner before handing text back to a caller like
        Storyteller, which is then searched for via
        ``get_locator_from_text`` -> ``find_text_location``. Without a
        joiner-aware exact/uniqueness match, a search phrase spanning an
        inline join can never satisfy ``full_text.find``/``count() == 1``,
        so it silently falls back to the normalized full-book scan -- which
        takes the FIRST normalized match, not a unique one. Here an
        ALL-CAPS decoy earlier in the book normalizes identically to the
        real (mixed-case) target, so a first-match fallback would return the
        wrong (earlier) location.
        """
        search_phrase = "The mysterious garden behind the old house held many secrets"
        content = (
            b"<html><body>"
            b"<p>THE MYSTERIOUS GARDEN BEHIND THE OLD HOUSE HELD MANY SECRETS "
            b"in an old story nobody remembers anymore.</p>"
            b"<p>Years later, someone returned. SENTINEL <b>Th</b>e mysterious garden "
            b"behind the old house held many secrets and nobody knew why.</p>"
            b"</body></html>"
        )
        item = SimpleNamespace(
            get_type=lambda: ITEM_DOCUMENT,
            get_content=lambda: content,
            get_name=lambda: "chapter.xhtml",
        )
        book = SimpleNamespace(
            spine=[("chapter", None)],
            get_item_with_id=lambda _item_id: item,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = Path(temp_dir) / "book.epub"
            epub_path.write_bytes(b"fixture")
            self.parser.books_dir = Path(temp_dir)
            self.parser.invalidate_path_cache()
            self.parser._build_href_resolver = lambda _path: lambda name: name
            with patch("src.utils.ebook_utils.epub.read_epub", return_value=book):
                full_text, _spine_map = self.parser.extract_text_and_map(epub_path)
                expected_index = full_text.index("SENTINEL") + len("SENTINEL ")
                self.assertEqual(
                    full_text[expected_index:expected_index + 3],
                    f"Th{INLINE_TEXT_JOINER}",
                )

                result = self.parser.find_text_location("book.epub", search_phrase)

        self.assertIsNotNone(result)
        self.assertEqual(result.match_index, expected_index)

    def test_content_guard_joins_inline_fragments_before_matching(self):
        visible_text = " ".join(f"word{i}" for i in range(200))
        fragmented_text = visible_text.replace("word", f"wo{EbookParser.INLINE_TEXT_JOINER}rd")
        service = AlignmentService.__new__(AlignmentService)
        service.ollama_client = None

        with patch.dict(
            os.environ,
            {
                "OLLAMA_ALIGN_CONTENT_GUARD": "true",
                "CONTENT_MATCH_GUARD": "true",
                "CONTENT_MATCH_MIN_OVERLAP": "0.15",
            },
        ):
            self.assertTrue(
                service._verify_content_match(
                    [{"start": 0.0, "end": 1.0, "text": visible_text}],
                    fragmented_text,
                    abs_id="inline-fixture",
                )
            )


if __name__ == "__main__":
    unittest.main()
