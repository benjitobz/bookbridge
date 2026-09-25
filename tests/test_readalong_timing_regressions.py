"""Regression coverage for segmented read-along timing assembly."""

import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.services.alignment_service import AlignmentService
from src.services.readalong_builder import _extend_clips_to_contiguous
from src.services.readalong_segments import SentenceClip, build_sentence_clips
from src.utils.ebook_utils import EbookParser
from src.utils.polisher import Polisher


_CONTAINER_XML = (
    '<?xml version="1.0"?><container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
    '<rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/></rootfiles></container>'
)


def _write_epub(path: Path) -> None:
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
        'version="3.0" unique-identifier="id"><metadata '
        'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title>'
        '<dc:identifier id="id">x</dc:identifier></metadata><manifest>'
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>'
        '</manifest><spine><itemref idref="ch1"/><itemref idref="ch2"/>'
        '</spine></package>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", _CONTAINER_XML)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/ch1.xhtml", b"<html><body><p>First sentence.</p></body></html>")
        archive.writestr("OEBPS/ch2.xhtml", b"<html><body><p>Second sentence.</p></body></html>")


def _write_one_spine_epub(path: Path) -> None:
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
        'version="3.0" unique-identifier="id"><metadata '
        'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title>'
        '<dc:identifier id="id">x</dc:identifier></metadata><manifest>'
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '</manifest><spine><itemref idref="ch1"/></spine></package>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", _CONTAINER_XML)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr(
            "OEBPS/ch1.xhtml",
            b"<html><body><p>First sentence. Second sentence.</p></body></html>",
        )


def _real_alignment_service(points, segments, total_chars):
    database = MagicMock()
    session = database.get_session.return_value
    session.__enter__.return_value = session
    row = SimpleNamespace(
        alignment_map_json=json.dumps(points),
        segments_json=json.dumps(segments),
    )
    session.query.return_value.filter_by.return_value.first.return_value = row
    database.get_alignment_total_chars.return_value = total_chars
    return AlignmentService(database, Polisher())


def test_real_alignment_reordered_spine_keeps_segment_bounds_and_audio_tail():
    """A later-narrated first chapter and earlier-narrated second chapter do
    not turn the second chapter's terminal clip into the whole audio tail."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        epub_path = root / "book.epub"
        books = root / "books"
        cache = root / "cache"
        books.mkdir()
        cache.mkdir()
        _write_epub(epub_path)
        parser = EbookParser(books_dir=str(books), epub_cache_dir=str(cache))
        combined_text, spine_map = parser.extract_text_and_map(str(epub_path))
        first, second = spine_map
        assert second["start"] == first["end"] + 1

        segments = [
            {"char_start": first["start"], "char_end": first["end"], "ts_start": 10.0, "ts_end": 20.0},
            {"char_start": second["start"], "char_end": second["end"], "ts_start": 0.0, "ts_end": 9.0},
        ]
        points = [
            {"char": first["start"], "ts": 10.0},
            {"char": first["end"] - 1, "ts": 20.0},
            {"char": second["start"], "ts": 0.0},
            {"char": second["end"] - 1, "ts": 9.0},
        ]
        service = _real_alignment_service(points, segments, len(combined_text))

        result = build_sentence_clips(parser, str(epub_path), service, "book")
        assert result is not None
        assert [(clip.ts_start, clip.ts_end) for clip in result.clips] == [(10.0, 20.0), (0.0, 9.0)]
        extended = _extend_clips_to_contiguous(result.clips, audio_duration_seconds=25.0)
        assert [(clip.ts_start, clip.ts_end) for clip in extended] == [(10.0, 25.0), (0.0, 9.0)]


def test_segmented_forward_jump_stops_at_own_boundary_and_latest_tail():
    """A forward reading-order jump cannot absorb the audio between blocks."""
    first = dict(sentence_id="first", spine_index=1, href="ch1.xhtml", char_start=0, char_end=1,
                 ts_start=0.0, ts_end=5.0, segment_key=1, segment_ts_start=0.0,
                 segment_ts_end=10.0, segment_scoped=True)
    second = dict(sentence_id="second", spine_index=2, href="ch2.xhtml", char_start=2, char_end=3,
                  ts_start=15.0, ts_end=18.0, segment_key=2, segment_ts_start=15.0,
                  segment_ts_end=20.0, segment_scoped=True)
    extended = _extend_clips_to_contiguous(
        [SentenceClip(**first), SentenceClip(**second)], audio_duration_seconds=20.0
    )
    assert [(clip.ts_start, clip.ts_end) for clip in extended] == [(0.0, 10.0), (15.0, 20.0)]


def test_three_way_reorder_does_not_overrun_audio_or_repeat_a_chapter():
    """Independent review's exact reproduction: naive reading-order-based
    contiguity extension (``ts_end[i] = ts_start[i+1]``, ignoring which
    segment each clip actually belongs to) can inflate the summed clip time
    past the real audio's own length and make two different chapters' <par>s
    claim overlapping real seconds -- "3 seconds of clips from 2 seconds of
    audio, repeating another chapter".

    Three single-sentence segments, each in its own 0.3s slot, read in an
    order that does not match narration order: chapter 1 (real audio
    0.0-0.3s) is read first, chapter 2 (real audio 1.5-1.8s) second, chapter
    3 (real audio 0.5-0.8s) last -- against 2.0s of real audio. The naive
    algorithm stretches chapter 1's clip forward to chapter 2's start (1.5s)
    and chapter 3's last clip out to the full audio duration (2.0s), so
    chapter 1's [0.0, 1.5) and chapter 3's [0.5, 2.0) both claim the real
    [0.5, 1.5) audio that only chapter 3 (per its own segment bounds) or
    nothing (unclaimed tail) actually owns -- three seconds of summed clip
    time (1.5 + 0 + 1.5) against two seconds of real audio, with the middle
    clamped to zero because its own extension went negative.
    """
    chapter_1 = dict(
        sentence_id="c1-s0", spine_index=1, href="ch1.xhtml", char_start=0, char_end=1,
        ts_start=0.0, ts_end=0.3, segment_key="seg1", segment_ts_start=0.0,
        segment_ts_end=0.3, segment_scoped=True,
    )
    chapter_2 = dict(
        sentence_id="c2-s0", spine_index=2, href="ch2.xhtml", char_start=2, char_end=3,
        ts_start=1.5, ts_end=1.8, segment_key="seg2", segment_ts_start=1.5,
        segment_ts_end=1.8, segment_scoped=True,
    )
    chapter_3 = dict(
        sentence_id="c3-s0", spine_index=3, href="ch3.xhtml", char_start=4, char_end=5,
        ts_start=0.5, ts_end=0.8, segment_key="seg3", segment_ts_start=0.5,
        segment_ts_end=0.8, segment_scoped=True,
    )
    original_by_key = {
        c["segment_key"]: (c["segment_ts_start"], c["segment_ts_end"])
        for c in (chapter_1, chapter_2, chapter_3)
    }
    clips = [SentenceClip(**chapter_1), SentenceClip(**chapter_2), SentenceClip(**chapter_3)]
    audio_duration = 2.0

    extended = _extend_clips_to_contiguous(clips, audio_duration_seconds=audio_duration)

    total_clip_time = sum(max(0.0, c.ts_end - c.ts_start) for c in extended)
    assert total_clip_time <= audio_duration, (
        f"summed clip time {total_clip_time}s exceeds the real audio's "
        f"{audio_duration}s"
    )

    for clip in extended:
        for other_key, (other_start, other_end) in original_by_key.items():
            if other_key == clip.segment_key:
                continue
            overlap_start = max(clip.ts_start, other_start)
            overlap_end = min(clip.ts_end, other_end)
            assert overlap_end <= overlap_start, (
                f"{clip.sentence_id} [{clip.ts_start}, {clip.ts_end}) repeats "
                f"segment {other_key}'s real audio [{other_start}, {other_end})"
            )


def test_real_alignment_reordered_sentences_within_one_spine_item_keep_both_ranges():
    """A reordered pair inside one XHTML file gets two independent overlay
    ranges instead of a single min/max span that replays the middle audio."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        epub_path = root / "book.epub"
        books = root / "books"
        cache = root / "cache"
        books.mkdir()
        cache.mkdir()
        _write_one_spine_epub(epub_path)
        parser = EbookParser(books_dir=str(books), epub_cache_dir=str(cache))
        combined_text, spine_map = parser.extract_text_and_map(str(epub_path))
        entry = spine_map[0]
        first_end = combined_text.index(".") + 1
        second_start = combined_text.index("Second")
        segments = [
            {"char_start": entry["start"], "char_end": first_end, "ts_start": 10.0, "ts_end": 20.0},
            {"char_start": second_start, "char_end": entry["end"], "ts_start": 0.0, "ts_end": 9.0},
        ]
        points = [
            {"char": entry["start"], "ts": 10.0},
            {"char": first_end - 1, "ts": 20.0},
            {"char": second_start, "ts": 0.0},
            {"char": entry["end"] - 1, "ts": 9.0},
        ]
        service = _real_alignment_service(points, segments, len(combined_text))

        result = build_sentence_clips(parser, str(epub_path), service, "book")
        assert result is not None
        extended = _extend_clips_to_contiguous(result.clips, audio_duration_seconds=25.0)
        assert [(clip.ts_start, clip.ts_end) for clip in extended] == [(10.0, 25.0), (0.0, 9.0)]
        assert sum(clip.ts_end - clip.ts_start for clip in extended) == 24.0
