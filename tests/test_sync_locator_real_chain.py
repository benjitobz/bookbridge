"""Round trips through the REAL sync chain, on real EPUB markup.

    audio timestamp -> canonical text offset -> reader locator
                    -> canonical text offset -> audio timestamp

`test_crossformat_drift_cycles.py` runs this loop against a fake parser and a
fake linear map, so it cannot see how real XHTML maps to a locator. These
tests use the real `EbookParser` locator methods (KOReader XPath and CFI) on
small EPUBs, each carrying one feature real books have: nonbreaking spaces,
inline formatting, duplicate phrases, chapter boundaries, a percent-encoded
path, and reordered narration through the real `AlignmentService` segment
logic.

Each round trip asserts position IDENTITY (same chapter, same paragraph, and
for a segmented map the same segment) as well as time. Time alone can hide a
wrong location: in a reordered book a small char error across a seam is hours
of audio, while a whole paragraph off can be a few seconds.

Tolerances. Both locator formats deliberately snap to the start of the
enclosing block (`.0` / `:0`, the crengine-safe rule in
`_build_crengine_safe_text_xpath`), so the exact return is the paragraph
start, not the word. The char bound is therefore "within the source
paragraph", and the time bound is the sync pipeline's own
`LOCATOR_ROUNDTRIP_TOLERANCE_SECONDS` default (30s): a round trip outside it
is one `_validate_and_stabilize_locator` would reject in production.
"""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

from src.services.alignment_service import AlignmentService
from src.utils.config_loader import DEFAULT_CONFIG
from src.utils.ebook_dom_map import block_break_offsets
from src.utils.ebook_utils import EbookParser
from src.utils.polisher import Polisher

_CONTAINER_XML = (
    '<?xml version="1.0"?><container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
    '<rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/></rootfiles></container>'
)

_ROUNDTRIP_TOLERANCE_SECONDS = float(DEFAULT_CONFIG["LOCATOR_ROUNDTRIP_TOLERANCE_SECONDS"])

# Real chapter shape: an XML declaration and XHTML namespace, as Calibre and
# every retail EPUB ship. ebooklib's reconstruction adds a doctype too.
_XHTML = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
    '<body>{body}</body></html>'
)


def _paragraphs(*texts: str) -> str:
    return "".join(f"<p>{text}</p>" for text in texts)


def _long(label: str) -> str:
    """A paragraph long enough that a position inside it is well past the
    ~40 chars an XML declaration plus doctype would add to a miscount."""
    return f"{label} opens this paragraph and keeps going with ordinary narrative words for a while."


FIXTURES: Dict[str, List[Tuple[str, str]]] = {
    "nbsp": [("c1.xhtml", _paragraphs(
        _long("First") + " Then nonbreaking spaces.",
        "  " + _long("Second"),
        _long("Third"),
    ))],
    "inline": [("c1.xhtml", _paragraphs(
        _long("First"),
        "Second <i>opens</i> this <b>paragraph <em>with</em></b> inline formatting and more words after it.",
        "Third <a href='#x'>paragraph</a> has a link inside it and several more ordinary words.",
    ))],
    "duplicate": [("c1.xhtml", _paragraphs(
        "The same line repeats here in this book, word for word, exactly.",
        _long("Middle"),
        "The same line repeats here in this book, word for word, exactly.",
        _long("Tail"),
    ))],
    "chapters": [
        ("c1.xhtml", _paragraphs(_long("Alpha"), _long("Bravo"))),
        ("c2.xhtml", _paragraphs(_long("Charlie"), _long("Delta"))),
        ("c3.xhtml", _paragraphs(_long("Echo"), _long("Foxtrot"))),
    ],
    "encoded": [("chapter%201.xhtml", _paragraphs(_long("Spaced"), _long("Another"), _long("Final")))],
    # Calibre's scene break, measured on Good Intentions: a CFI into the
    # 5-char "* * *" span resolved to the chapter START (77,117 chars back).
    "scene_break": [("c1.xhtml", (
        _paragraphs(_long("First"), _long("Second"))
        + '<p class="calibre2"><span class="calibre5">* * *</span></p>'
        + _paragraphs(_long("Third"), _long("Fourth"))
    ))],
}


def _write_epub(path: Path, items: List[Tuple[str, str]]) -> None:
    manifest = "".join(
        f'<item id="i{n}" href="{href}" media-type="application/xhtml+xml"/>'
        for n, (href, _body) in enumerate(items)
    )
    spine = "".join(f'<itemref idref="i{n}"/>' for n in range(len(items)))
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
        'version="2.0" unique-identifier="id"><metadata '
        'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title>'
        '<dc:identifier id="id">x</dc:identifier></metadata>'
        f'<manifest>{manifest}</manifest><spine>{spine}</spine></package>'
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", _CONTAINER_XML)
        z.writestr("OEBPS/content.opf", opf)
        for href, body in items:
            z.writestr("OEBPS/" + href.replace("%20", " "), _XHTML.format(body=body))


def _alignment_service(alignment_map: List[Dict], segments: Optional[List[Dict]] = None) -> AlignmentService:
    """A real AlignmentService over a stubbed DB row (the pattern in
    tests/test_segmented_map.py), so segment-aware interpolation is real."""
    mock_db = MagicMock()
    session = mock_db.get_session()
    session.__enter__.return_value = session
    entry = MagicMock()
    entry.alignment_map_json = json.dumps(alignment_map)
    entry.segments_json = json.dumps(segments) if segments is not None else None
    session.query.return_value.filter_by.return_value.first.return_value = entry
    return AlignmentService(mock_db, Polisher())


class _RealChainCase(unittest.TestCase):
    """Shared scaffolding: one EPUB per fixture, parsed by a real EbookParser."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        (tmp / "books").mkdir()
        (tmp / "cache").mkdir()
        self.books = tmp / "books"
        self.parser = EbookParser(books_dir=str(tmp / "books"), epub_cache_dir=str(tmp / "cache"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _book(self, name: str) -> Tuple[str, str, List[Dict]]:
        filename = f"{name}.epub"
        _write_epub(self.books / filename, FIXTURES[name])
        text, spine_map = self.parser.extract_text_and_map(filename)
        return filename, text, spine_map

    @staticmethod
    def _paragraph_starts(text: str, spine_map: List[Dict]) -> List[int]:
        """Canonical start offset of every block element, from the same DOM
        mapping the read-along builder uses (`block_break_offsets`)."""
        starts = []
        for item in spine_map:
            chapter = text[item["start"]:item["end"]]
            starts.append(item["start"])
            breaks = block_break_offsets(item["content"], chapter) or []
            starts.extend(item["start"] + b for b in breaks)
        return sorted(set(starts))

    @staticmethod
    def _paragraph_of(offset: int, paragraph_starts: List[int]) -> int:
        return max(i for i, start in enumerate(paragraph_starts) if start <= offset)

    def _acceptable_paragraphs(self, text: str, offset: int, paragraph_starts: List[int],
                               substitutes: Optional[Dict[int, int]] = None) -> set:
        """Paragraphs a correct round trip may land in. An offset ON the
        separator space between two blocks belongs to neither, so both
        neighbours are correct. `substitutes` maps a paragraph to the one a
        KOReader XPath deliberately lands on instead: for a <p> with
        fragmenting inline children, `_build_crengine_safe_text_xpath`
        substitutes the nearest clean <p> sibling (a crengine compatibility
        rule this test preserves, not judges)."""
        own = self._paragraph_of(offset, paragraph_starts)
        accepted = {own}
        if offset < len(text) and text[offset].isspace():
            accepted.add(self._paragraph_of(offset + 1, paragraph_starts))
        if substitutes:
            accepted |= {substitutes[p] for p in list(accepted) if p in substitutes}
        return accepted

    @staticmethod
    def _spine_of(offset: int, spine_map: List[Dict]) -> Optional[int]:
        for item in spine_map:
            if item["start"] <= offset < item["end"]:
                return item["spine_index"]
        return None

    def _locator_round_trips(self, filename: str, offset: int) -> Dict[str, Optional[int]]:
        locator = self.parser.get_locator_from_char_offset(filename, offset)
        self.assertIsNotNone(locator, f"no locator for offset {offset}")
        return {
            "xpath": self.parser.resolve_xpath_to_index(filename, locator.perfect_ko_xpath),
            "cfi": self.parser.resolve_cfi_to_index(filename, locator.cfi),
        }


class TestLocatorRoundTripIdentity(_RealChainCase):
    """char -> locator -> char lands in the SAME paragraph and chapter, at its
    start (the deliberate block-level snap)."""

    def _assert_identity(self, name: str, formats: Tuple[str, ...] = ("xpath", "cfi"),
                         xpath_substitutes: Optional[Dict[int, int]] = None) -> None:
        filename, text, spine_map = self._book(name)
        paragraph_starts = self._paragraph_starts(text, spine_map)
        # Probe mid-paragraph (well past the opening) and at the paragraph start.
        for start in paragraph_starts:
            end = min([s for s in paragraph_starts if s > start] + [len(text)])
            for offset in (start, (start + end) // 2):
                back = self._locator_round_trips(filename, offset)
                for fmt in formats:
                    with self.subTest(fixture=name, fmt=fmt, offset=offset, text=text[offset:offset + 20]):
                        self.assertIsNotNone(back[fmt])
                        self.assertEqual(self._spine_of(back[fmt], spine_map), self._spine_of(offset, spine_map))
                        self.assertIn(
                            self._paragraph_of(back[fmt], paragraph_starts),
                            self._acceptable_paragraphs(
                                text, offset, paragraph_starts,
                                substitutes=xpath_substitutes if fmt == "xpath" else None,
                            ),
                            f"{fmt} round trip left the paragraph: {offset} -> {back[fmt]} "
                            f"({text[back[fmt]:back[fmt] + 20]!r})",
                        )

    def test_nonbreaking_spaces(self) -> None:
        self._assert_identity("nbsp")

    def test_inline_formatting(self) -> None:
        """CFI keeps the exact paragraph. Paragraphs 2 and 3 both carry
        inline markup, so the KOReader XPath lands on paragraph 1, the nearest
        clean sibling, by design (see `_acceptable_paragraphs`)."""
        self._assert_identity("inline", xpath_substitutes={1: 0, 2: 0})

    def test_chapter_boundaries(self) -> None:
        self._assert_identity("chapters")

    def test_percent_encoded_chapter_path(self) -> None:
        self._assert_identity("encoded")

    def test_duplicate_phrases(self) -> None:
        """The second copy of a repeated line must not resolve to the first:
        both resolvers fall back to position when the text is ambiguous."""
        self._assert_identity("duplicate")

    def test_scene_break(self) -> None:
        """A CFI into a short scene-break span resolves to the break, not to
        the chapter start. The break's <p> wraps a <span>, so its KOReader
        XPath takes the previous clean paragraph (index 1) by design."""
        self._assert_identity("scene_break", xpath_substitutes={2: 1})


class TestAudioRoundTripThroughLocators(_RealChainCase):
    """The full chain with a real AlignmentService: ts -> char -> locator ->
    char -> ts stays in the same paragraph and inside the sync tolerance."""

    CHARS_PER_SECOND = 15.0  # typical narration speed

    def test_linear_map_round_trip_through_both_locator_formats(self) -> None:
        filename, text, spine_map = self._book("chapters")
        service = _alignment_service([
            {"char": 0, "ts": 0.0},
            {"char": len(text), "ts": len(text) / self.CHARS_PER_SECOND},
        ])
        paragraph_starts = self._paragraph_starts(text, spine_map)
        duration = len(text) / self.CHARS_PER_SECOND
        for step in range(1, 12):
            ts = duration * step / 12
            offset = service.get_char_for_time("b", ts)
            back = self._locator_round_trips(filename, offset)
            for fmt, back_offset in back.items():
                with self.subTest(fmt=fmt, ts=ts):
                    self.assertIn(
                        self._paragraph_of(back_offset, paragraph_starts),
                        self._acceptable_paragraphs(text, offset, paragraph_starts),
                    )
                    self.assertLessEqual(
                        abs(service.get_time_for_char("b", back_offset) - ts),
                        _ROUNDTRIP_TOLERANCE_SECONDS,
                    )


class TestReorderedNarrationRoundTrip(_RealChainCase):
    """Out-of-order narration (issue #426): chapter 3 is narrated first. A
    round trip must come back in the SAME segment -- a wrong segment is hours
    of audio even when the char error is tiny."""

    def _segmented(self) -> Tuple[str, str, List[Dict], AlignmentService, List[Dict]]:
        filename, text, spine_map = self._book("chapters")
        ch1, ch2, ch3 = spine_map
        rate = 15.0
        ch3_len = ch3["end"] - ch3["start"]
        ch12_len = ch2["end"] - ch1["start"]
        # Narration order: chapter 3 first (0 .. ch3_len/rate), then 1-2.
        segments = [
            {"char_start": ch3["start"], "char_end": ch3["end"],
             "ts_start": 0.0, "ts_end": ch3_len / rate},
            {"char_start": ch1["start"], "char_end": ch2["end"],
             "ts_start": ch3_len / rate, "ts_end": (ch3_len + ch12_len) / rate},
        ]
        flat_map = sorted([
            {"char": ch1["start"], "ts": ch3_len / rate},
            {"char": ch2["end"] - 1, "ts": (ch3_len + ch12_len) / rate},
            {"char": ch3["start"], "ts": 0.0},
            {"char": ch3["end"] - 1, "ts": ch3_len / rate},
        ], key=lambda p: p["char"])
        return filename, text, spine_map, _alignment_service(flat_map, segments), segments

    @staticmethod
    def _segment_of(offset: int, segments: List[Dict]) -> Optional[int]:
        for i, segment in enumerate(segments):
            if segment["char_start"] <= offset < segment["char_end"]:
                return i
        return None

    def test_round_trip_keeps_the_segment_and_the_time(self) -> None:
        filename, text, spine_map, service, segments = self._segmented()
        total = max(s["ts_end"] for s in segments)
        for step in range(1, 20):
            ts = total * step / 20
            offset = service.get_char_for_time("b", ts)
            back = self._locator_round_trips(filename, offset)
            for fmt, back_offset in back.items():
                with self.subTest(fmt=fmt, ts=ts, offset=offset):
                    self.assertEqual(self._segment_of(back_offset, segments), self._segment_of(offset, segments))
                    self.assertLessEqual(
                        abs(service.get_time_for_char("b", back_offset) - ts),
                        _ROUNDTRIP_TOLERANCE_SECONDS,
                    )

    def test_offset_in_no_segment_resolves_to_a_segment_edge(self) -> None:
        """Pins the intentional approximation: a char no segment covers (the
        separator between chapters here) answers with the nearest segment
        edge's time, not an interpolation across the seam."""
        _filename, _text, spine_map, service, segments = self._segmented()
        gap_offset = spine_map[0]["start"] - 1 if spine_map[0]["start"] > 0 else None
        if gap_offset is None:
            gap_offset = spine_map[2]["start"] - 1
        edge_times = {s["ts_start"] for s in segments} | {s["ts_end"] for s in segments}
        self.assertIn(service.get_time_for_char("b", gap_offset), edge_times)


class TestOtherLocatorPathsUseCanonicalText(_RealChainCase):
    """The same miscount -- counting ebooklib's <?xml?> declaration and
    doctype as text, and (for fragments) no separator space between text
    nodes -- lived in two more locator paths."""

    def test_fragment_id_resolves_to_the_element_text(self) -> None:
        """Storyteller/Readium fragment -> text snippet. The snippet the sync
        path fuzzy-matches must start at the element, not ~40 chars off."""
        FIXTURES["fragments"] = [("c1.xhtml", (
            _paragraphs(_long("First"), _long("Second"), _long("Third"))
            + '<p id="target-para">Target paragraph opens right here with its own words.</p>'
            + _paragraphs(_long("Fifth"))
        ))]
        try:
            filename, _text, spine_map = self._book("fragments")
            snippet = self.parser.resolve_locator_id(filename, spine_map[0]["href"], "target-para")
        finally:
            del FIXTURES["fragments"]
        self.assertIsNotNone(snippet)
        self.assertTrue(
            snippet.startswith("Target paragraph opens right here"),
            f"fragment resolved to the wrong offset: {snippet[:60]!r}",
        )

    def test_fuzzy_match_locator_targets_the_matched_paragraph(self) -> None:
        """find_text_location's rich locator (XPath/CSS via _generate_xpath_bs4)
        must point at the paragraph the text matched, not an earlier one."""
        filename, text, spine_map = self._book("chapters")
        locator = self.parser.find_text_location(filename, "Bravo opens this paragraph and keeps going")
        self.assertIsNotNone(locator)
        item = next(i for i in spine_map if i["start"] <= locator.match_index < i["end"])
        _xpath, target_tag, _anchored = self.parser._generate_xpath_bs4(
            item["content"], locator.match_index - item["start"],
        )
        self.assertIsNotNone(target_tag)
        self.assertTrue(
            target_tag.get_text().startswith("Bravo opens"),
            f"fuzzy locator targeted {target_tag.get_text()[:40]!r}",
        )


if __name__ == "__main__":
    unittest.main()
