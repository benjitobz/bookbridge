"""Tests for `src/services/ctc_search.py`.

No audio and no torch: a synthetic "decoded document" stands in for a greedy
CTC decoding. `_build_synthetic_document` lays out known chapter texts (run
through `build_query`, exactly as a real caller would) at a fixed frame rate
with silence gaps between chapters and deterministic substitution/drop noise,
and records each chapter's *ground-truth* frame for every one of its query
characters (computed inline, independent of whether that character survives
the noise) so tests can check located boundaries and anchors against an exact
target rather than an approximation.
"""

from __future__ import annotations

import random
import time
from typing import List, NamedTuple, Tuple

import numpy as np

from src.services.ctc_search import (
    ChapterSearchResult,
    NGRAM_SIZE,
    MAX_SEARCH_LENGTH,
    BoundaryMatch,
    PositionedDocument,
    build_query,
    find_boundaries,
    greedy_decode,
    search_chapters,
)

_ALPHABET = "abcdefghijklmnopqrstuvwxyz'"
_CONSONANTS = "bcdfghjklmnpqrstvwxyz"
_VOWELS = "aeiou"

# The padding rule in `_find_boundaries_in_document`: each edge gets at least
# `BOUNDARY_PAD_FRAMES` (500) of pad, plus up to 25% of the extrapolation
# overrun beyond the first/last refined match. With our synthetic ~3
# frames/char rate and light per-char timing jitter (0-1 extra frame,
# accumulating slowly), the dynamic term stays small next to the fixed 500,
# so located edges land within about 2x that pad of the true edge.
_BOUNDARY_TOLERANCE_FRAMES = 1200

# Queries in these tests are well under 15,000 chars, so `find_boundaries`
# uses the smaller of the two `findCtcBoundaries` inlier tolerances (2500,
# not 5000) -- anchors must land within that of their true frame.
_INLIER_TOLERANCE = 2500


def _make_prose(rng: random.Random, n_chars: int) -> str:
    """Deterministic pseudo-prose: consonant/vowel-alternating "words" of
    varied length, separated by spaces, with an occasional apostrophe
    contraction -- enough lexical variety that 10-grams are almost all
    unique, like real prose, without needing a real corpus."""
    words: List[str] = []
    total = 0
    i = 0
    while total < n_chars:
        wlen = rng.randint(2, 9)
        letters = []
        for j in range(wlen):
            letters.append(rng.choice(_CONSONANTS) if j % 2 == 0 else rng.choice(_VOWELS))
        word = "".join(letters)
        if i % 17 == 0 and wlen > 3:
            word = word[:-2] + "'" + word[-2:]
        words.append(word)
        total += wlen + 1
        i += 1
    return " ".join(words)


class SyntheticBook(NamedTuple):
    document: PositionedDocument
    num_frames: int
    true_ranges: List[Tuple[int, int]]
    """[(start_frame, end_frame)] per chapter, matching each chapter's own
    `build_query` output span in the decoded document (before boundary
    padding)."""
    true_frame_for_offset: List[np.ndarray]
    """Per chapter, per query-offset ground-truth frame -- the frame that
    offset *would* occupy whether or not noise dropped it."""
    queries: List[str]
    """Per chapter, the normalized (`build_query`) text used to build the
    document."""


def _build_synthetic_document(
    chapter_texts: List[str],
    rng: random.Random,
    frames_per_char: int = 3,
    noise_sub_rate: float = 0.05,
    noise_drop_rate: float = 0.03,
    gap_frames: int = 2000,
) -> SyntheticBook:
    doc_chars: List[str] = []
    doc_positions: List[int] = []
    true_ranges: List[Tuple[int, int]] = []
    true_frame_for_offset: List[np.ndarray] = []
    queries: List[str] = []

    t = 0
    for text in chapter_texts:
        query, _ = build_query(text, 0, len(text))
        queries.append(query)

        start_t = t
        offsets_frames = np.empty(len(query), dtype=np.int64)
        for o, ch in enumerate(query):
            offsets_frames[o] = t
            if rng.random() >= noise_drop_rate:
                emit_ch = rng.choice(_ALPHABET) if rng.random() < noise_sub_rate else ch
                doc_chars.append(emit_ch)
                doc_positions.append(t)
            t += frames_per_char + rng.randint(0, 1)
        end_t = t

        true_ranges.append((start_t, end_t))
        true_frame_for_offset.append(offsets_frames)
        t += gap_frames

    document = PositionedDocument(text="".join(doc_chars), positions=np.array(doc_positions, dtype=np.int64))
    return SyntheticBook(
        document=document,
        num_frames=t,
        true_ranges=true_ranges,
        true_frame_for_offset=true_frame_for_offset,
        queries=queries,
    )


def _spans_in(full_text: str, chapter_texts: List[str], sep: str = "\n\n") -> List[Tuple[int, int]]:
    """(start_char, end_char) for each chapter_texts entry as concatenated
    into full_text with `sep` between them."""
    spans = []
    pos = 0
    for text in chapter_texts:
        spans.append((pos, pos + len(text)))
        pos += len(text) + len(sep)
    assert full_text == sep.join(chapter_texts)
    return spans


# --------------------------------------------------------------------------- #
# greedy_decode
# --------------------------------------------------------------------------- #


def test_greedy_decode_collapses_repeats_and_skips_blank():
    # tokens: 1=a 2=b 3=c 4=d, 0=blank. Frame sequence chosen to exercise:
    # repeats collapsed (1,1 -> one 'a'), blank skipped (no char emitted for
    # 0), and the SAME letter ('c') appearing again after being separated by
    # a blank frame -- two separate emissions, not merged into one.
    tokens = [1, 1, 2, 2, 2, 0, 0, 1, 1, 3, 3, 3, 0, 3, 4, 4, 0, 0]
    #         a  a  b  b  b  -  -  a  a  c  c  c  -  c  d  d  -  -
    # argmax changes at: t0(->a), t2(->b), t5(->blank,skip), t7(->a again),
    # t9(->c), t12(->blank,skip), t13(->c again), t14(->d), t16(->blank,skip)
    T, C = len(tokens), 5
    log_probs = np.full((T, C), -10.0)
    for t, tok in enumerate(tokens):
        log_probs[t, tok] = 0.0

    id_to_char = {1: "a", 2: "b", 3: "c", 4: "d"}
    text, frames = greedy_decode(log_probs, blank_id=0, id_to_char=id_to_char)

    assert text == "abaccd"
    assert frames.tolist() == [0, 2, 7, 9, 13, 14]
    assert frames.dtype == np.int64


def test_greedy_decode_exact_sequence():
    """A simpler, fully hand-traced case so the expected text/frames are
    unambiguous."""
    # token stream, ids: blank=0, a=1, b=2, c=3
    tokens = [1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 3, 1]
    #         a  a  a  -  -  b  b  -  a  a  c  a
    # argmax changes at: t0(1,new->a), t3(0,blank skip),
    # t5(2,new->b), t7(0,blank skip), t8(1,new->a again),
    # t10(3,new->c), t11(1,new->a again)
    T, C = len(tokens), 4
    log_probs = np.full((T, C), -10.0)
    for t, tok in enumerate(tokens):
        log_probs[t, tok] = 0.0
    id_to_char = {1: "a", 2: "b", 3: "c"}

    text, frames = greedy_decode(log_probs, blank_id=0, id_to_char=id_to_char)

    assert text == "abaca"
    assert frames.tolist() == [0, 5, 8, 10, 11]


def test_greedy_decode_skips_tokens_missing_from_id_to_char():
    # token 2 has no entry in id_to_char (stands in for e.g. "|" or "*"),
    # so its runs never emit even though they change the argmax.
    tokens = [1, 1, 2, 2, 1, 1]
    T, C = len(tokens), 3
    log_probs = np.full((T, C), -10.0)
    for t, tok in enumerate(tokens):
        log_probs[t, tok] = 0.0
    id_to_char = {1: "a"}

    text, frames = greedy_decode(log_probs, blank_id=0, id_to_char=id_to_char)

    assert text == "aa"
    assert frames.tolist() == [0, 4]


def test_greedy_decode_empty_input():
    log_probs = np.zeros((0, 3))
    text, frames = greedy_decode(log_probs, blank_id=0, id_to_char={1: "a"})
    assert text == ""
    assert frames.tolist() == []


# --------------------------------------------------------------------------- #
# build_query
# --------------------------------------------------------------------------- #


def test_build_query_normalizes_and_maps_offsets():
    full_text = "Hello, World! It's a café-visit 2024."
    query, offsets = build_query(full_text, 0, len(full_text))

    assert query == "helloworldit'sacafvisit"
    assert len(offsets) == len(query)
    for char, offset in zip(query, offsets):
        assert full_text[offset].lower() == char
    # apostrophe preserved
    assert "'" in query
    # digits and punctuation dropped entirely (not replaced by anything)
    assert not any(c.isdigit() for c in query)
    assert "," not in query and "!" not in query and "-" not in query
    # accented char (e in café) dropped, not folded to plain "e"
    assert "cafe" not in query
    assert "caf" in query


def test_build_query_respects_start_end_bounds():
    full_text = "abc DEF ghi"
    query, offsets = build_query(full_text, 4, 7)
    assert query == "def"
    assert offsets == [4, 5, 6]


def test_build_query_empty_range():
    query, offsets = build_query("hello", 2, 2)
    assert query == ""
    assert offsets == []


# --------------------------------------------------------------------------- #
# find_boundaries
# --------------------------------------------------------------------------- #


def test_find_boundaries_locates_each_chapter_within_tolerance():
    rng = random.Random(1001)
    chapter_texts = [_make_prose(rng, 3200) for _ in range(3)]
    book = _build_synthetic_document(chapter_texts, rng)

    for i, query in enumerate(book.queries):
        result = find_boundaries(query, book.document, [], num_frames=book.num_frames)
        assert result is not None, f"chapter {i} not found"

        true_start, true_end = book.true_ranges[i]
        assert abs(result.start - true_start) <= _BOUNDARY_TOLERANCE_FRAMES, (
            i, result.start, true_start,
        )
        assert abs(result.end - true_end) <= _BOUNDARY_TOLERANCE_FRAMES, (
            i, result.end, true_end,
        )

        assert result.confidence >= 0.5

        # anchors: monotonic in both coordinates, each within the inlier
        # tolerance of its ground-truth frame for that query offset.
        true_frames = book.true_frame_for_offset[i]
        assert len(result.anchors) >= 1
        prev_offset, prev_frame = -1, -1
        for offset, frame in result.anchors:
            assert offset > prev_offset
            assert frame >= prev_frame
            prev_offset, prev_frame = offset, frame
            assert abs(frame - int(true_frames[offset])) <= _INLIER_TOLERANCE


def test_find_boundaries_absent_chapter_returns_none():
    rng = random.Random(2002)
    chapter_texts = [_make_prose(rng, 3000) for _ in range(2)]
    book = _build_synthetic_document(chapter_texts, rng)

    absent_text = _make_prose(random.Random(99999), 3000)
    absent_query, _ = build_query(absent_text, 0, len(absent_text))

    result = find_boundaries(absent_query, book.document, [], num_frames=book.num_frames)
    assert result is None


def test_find_boundaries_rejects_opening_only_match_by_coverage():
    """A query whose first ~20% is real chapter text and whose remaining
    ~80% is unrelated should be rejected (MIN_COVERAGE = 0.3, i.e. matched
    span must cover at least 30% of the query)."""
    rng = random.Random(3003)
    chapter_texts = [_make_prose(rng, 3200) for _ in range(1)]
    book = _build_synthetic_document(chapter_texts, rng)

    real_query = book.queries[0]
    cutoff = int(len(real_query) * 0.2)
    unrelated_text = _make_prose(random.Random(4004), 8000)
    unrelated_query, _ = build_query(unrelated_text, 0, len(unrelated_text))
    spliced_query = real_query[:cutoff] + unrelated_query[: len(real_query) - cutoff]

    result = find_boundaries(spliced_query, book.document, [], num_frames=book.num_frames)
    assert result is None


def test_find_boundaries_is_deterministic():
    rng = random.Random(5005)
    chapter_texts = [_make_prose(rng, 3200) for _ in range(2)]
    book = _build_synthetic_document(chapter_texts, rng)

    query = book.queries[0]
    result_a = find_boundaries(query, book.document, [], num_frames=book.num_frames)
    result_b = find_boundaries(query, book.document, [], num_frames=book.num_frames)

    assert result_a is not None and result_b is not None
    assert result_a == result_b


# --------------------------------------------------------------------------- #
# search_chapters
# --------------------------------------------------------------------------- #


def test_search_chapters_in_order_all_found_ascending_nonoverlapping():
    rng = random.Random(6006)
    chapter_texts = [_make_prose(rng, 3200) for _ in range(3)]
    book = _build_synthetic_document(chapter_texts, rng)

    full_text = "\n\n".join(chapter_texts)
    chapters = _spans_in(full_text, chapter_texts)

    results = search_chapters(book.document, full_text, chapters, book.num_frames)

    assert len(results) == 3
    for r in results:
        assert r.found
        assert r.start_frame is not None and r.end_frame is not None
        assert r.confidence is not None and r.confidence >= 0.5

    # ascending, non-overlapping in spine order
    assert results[0].end_frame < results[1].start_frame
    assert results[1].end_frame < results[2].start_frame


def test_search_chapters_out_of_order_narration_all_found_not_ascending():
    rng = random.Random(7007)
    chapter_texts = [_make_prose(rng, 3200) for _ in range(3)]
    # narrated order: chapter[2], chapter[0], chapter[1]
    narrated_order = [chapter_texts[2], chapter_texts[0], chapter_texts[1]]
    book = _build_synthetic_document(narrated_order, rng)

    full_text = "\n\n".join(chapter_texts)
    chapters = _spans_in(full_text, chapter_texts)  # spine order: 0, 1, 2

    results = search_chapters(book.document, full_text, chapters, book.num_frames)

    assert len(results) == 3
    for r in results:
        assert r.found

    # Each is still located...
    # ...but since chapter[2] was narrated FIRST, its frame range precedes
    # chapter[0]'s and chapter[1]'s -- NOT ascending in spine order. The
    # caller (Phase 2) decides what to do with that; this only asserts the
    # fact.
    assert results[2].start_frame < results[0].start_frame
    assert results[2].start_frame < results[1].start_frame
    assert not (
        results[0].end_frame < results[1].start_frame < results[1].end_frame < results[2].start_frame
    )


def test_search_chapters_excludes_previously_claimed_range():
    """A chapter duplicated verbatim twice in the document: searching it as
    two spine chapters finds two DIFFERENT, non-overlapping occurrences --
    the second search excludes the range the first one claimed."""
    rng = random.Random(8008)
    chapter_text = _make_prose(rng, 3200)
    book = _build_synthetic_document([chapter_text, chapter_text], rng)

    # Two spine "chapters" pointing at the same source text (as if the same
    # passage were duplicated in the book).
    full_text = chapter_text + "\n\n" + chapter_text
    chapters = [(0, len(chapter_text)), (len(chapter_text) + 2, len(chapter_text) + 2 + len(chapter_text))]

    results = search_chapters(book.document, full_text, chapters, book.num_frames)

    assert len(results) == 2
    assert results[0].found and results[1].found

    a_start, a_end = results[0].start_frame, results[0].end_frame
    b_start, b_end = results[1].start_frame, results[1].end_frame
    # no overlap between the two claimed ranges
    assert a_end < b_start or b_end < a_start

    # and together they correspond to the book's two true occurrence windows
    # (order-independent: whichever copy each search happened to land on).
    true_a, true_b = book.true_ranges
    found_spans = sorted([(a_start, a_end), (b_start, b_end)])
    true_spans = sorted([true_a, true_b])
    for (fs, fe), (ts, te) in zip(found_spans, true_spans):
        assert abs(fs - ts) <= _BOUNDARY_TOLERANCE_FRAMES
        assert abs(fe - te) <= _BOUNDARY_TOLERANCE_FRAMES


def test_search_chapters_chunks_a_chapter_over_40k_chars():
    rng = random.Random(9009)
    # Comfortably over 40,000 chars *after* normalization (build_query strips
    # spaces, so generate more raw chars than the threshold).
    big_text = _make_prose(rng, 58000)
    query, _ = build_query(big_text, 0, len(big_text))
    assert len(query) > MAX_SEARCH_LENGTH, "test fixture must exceed the chunking threshold"

    book = _build_synthetic_document([big_text], rng)
    results = search_chapters(book.document, big_text, [(0, len(big_text))], book.num_frames)

    assert len(results) == 1
    result = results[0]
    assert result.found
    true_start, true_end = book.true_ranges[0]
    assert abs(result.start_frame - true_start) <= _BOUNDARY_TOLERANCE_FRAMES
    assert abs(result.end_frame - true_end) <= _BOUNDARY_TOLERANCE_FRAMES
    # anchors from more than one chunk should be present for a chapter this
    # long (ANCHOR_SPACING=2000 over >40k chars per chunk => >1 anchor/chunk,
    # and 2 chunks total given the 40k threshold).
    assert len(result.anchors) > 13


def test_search_chapters_too_short_chapter_not_found():
    rng = random.Random(10010)
    chapter_texts = [_make_prose(rng, 3200), "Hi."]  # second chapter: far under 2*NGRAM_SIZE chars
    book = _build_synthetic_document(chapter_texts, rng)

    full_text = "\n\n".join(chapter_texts)
    chapters = _spans_in(full_text, chapter_texts)

    results = search_chapters(book.document, full_text, chapters, book.num_frames)

    assert results[0].found
    assert results[1].found is False
    assert results[1].start_frame is None
    assert results[1].end_frame is None
    assert results[1].anchors == []


def test_search_chapters_is_deterministic():
    rng = random.Random(11011)
    chapter_texts = [_make_prose(rng, 3200) for _ in range(3)]
    book = _build_synthetic_document(chapter_texts, rng)
    full_text = "\n\n".join(chapter_texts)
    chapters = _spans_in(full_text, chapter_texts)

    results_a = search_chapters(book.document, full_text, chapters, book.num_frames)
    results_b = search_chapters(book.document, full_text, chapters, book.num_frames)

    assert results_a == results_b


# --------------------------------------------------------------------------- #
# Performance guard
# --------------------------------------------------------------------------- #


def test_search_chapters_performance_on_150k_char_book():
    """A 150k-char synthetic book (5 chapters, ~30k chars each) should search
    in well under the generous 20s budget on typical dev hardware."""
    rng = random.Random(12012)
    chapter_texts = [_make_prose(rng, 30000) for _ in range(5)]
    book = _build_synthetic_document(chapter_texts, rng)
    full_text = "\n\n".join(chapter_texts)
    chapters = _spans_in(full_text, chapter_texts)

    start = time.perf_counter()
    results = search_chapters(book.document, full_text, chapters, book.num_frames)
    elapsed = time.perf_counter() - start

    print(f"\nsearch_chapters on 150k-char/5-chapter synthetic book: {elapsed:.2f}s")

    assert len(results) == 5
    for r in results:
        assert r.found
    assert elapsed < 20.0
