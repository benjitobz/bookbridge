"""CTC chapter search: locate spine chapters inside a greedy-decoded CTC stream.

Port of Storyteller's ``libraries/align/src/align/ctc/search.ts`` and
``libraries/align/src/align/ctc/greedyDecode.ts`` (web-v3.0.0-beta.38, MIT
License, Copyright (c) 2023 Shane Friedman), plus the book-level chunking/
exclusion concepts from ``libraries/align/src/align/ctc/Aligner.ts``
(``MAX_SEARCH_LENGTH`` chunking, ``narrowToAvailableBoundary``,
``getMatchedBoundaries``). Nothing from ``libraries/ghost-story`` (GPL-3.0) is
used or needed -- our emissions come from torchaudio MMS_FA, not Storyteller's
CTC engine.

Given a CTC model's emissions, ``greedy_decode`` collapses them into a rough,
space-free character stream where every character carries the emission frame
it was first observed at. ``find_boundaries`` then locates a query string (a
normalized chapter of book text) inside that stream via n-gram matching +
RANSAC line-fitting over (query offset, document frame) pairs, tolerating the
CTC stream's substitution/insertion/deletion noise. ``search_chapters`` runs
that over a whole spine in order, excluding frame ranges already claimed by
earlier chapters so a repeated or duplicated passage cannot be double-matched.

This module has no dependency on torch/torchaudio and does not import them --
``log_probs`` is handed in as a plain ``numpy`` array by the caller.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

__all__ = [
    "PositionedDocument",
    "BoundaryMatch",
    "ChapterSearchResult",
    "greedy_decode",
    "build_query",
    "find_boundaries",
    "search_chapters",
    "compute_evidence_range",
]

# --------------------------------------------------------------------------- #
# Constants -- names and values match ctc_search.ts / ctc_greedyDecode.ts /
# ctc_Aligner.ts exactly (see the module docstring for provenance).
# --------------------------------------------------------------------------- #

NGRAM_SIZE = 10
RANSAC_ITERATIONS = 500
MIN_INLIERS = 10
MAX_PAIR_SEPARATION_REQUIREMENT = 200
MIN_CONFIDENCE = 0.5
MIN_COVERAGE = 0.3
REFINE_COUNT = 25
ANCHOR_SPACING = 2000
BOUNDARY_PAD_FRAMES = 500
EVIDENCE_GAP_MULTIPLIER = 3
EVIDENCE_GAP_FLOOR = 500
EVIDENCE_GAP_PERCENTILE = 0.99
MIN_EDGE_CLUSTER_OFFSETS = 10
EDGE_EXTENSION_MAX_GAP = 1000
EDGE_EXTENSION_WINDOW = 250
EDGE_EXTENSION_PAIR_SEPARATION = 50
ANCHOR_CONTEXT = 40
ANCHOR_AGREEMENT_FRACTION = 0.4
ANCHOR_AGREEMENT_FLOOR = 0.05
ANCHOR_CALIBRATION_MINIMUM = 5

# From ctc_Aligner.ts -- the book-level chunking threshold for a single chapter's
# search query.
MAX_SEARCH_LENGTH = 40_000

# The forced-align target character rule (mirrors _MMS_CHARS_RE in
# src/utils/forced_aligner.py: lowercase latin + apostrophe only). Not imported
# from that module per the porting brief -- replicated here deliberately.
_QUERY_KEEP_RE = re.compile(r"[a-z']")


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


class Match(NamedTuple):
    """One n-gram match: ``offset`` into the query, ``position`` in the document."""

    offset: int
    position: int


class _LineFit(NamedTuple):
    slope: float
    intercept: float
    inliers: List[Match]


@dataclass(eq=False)
class PositionedDocument:
    """A searchable decoded stream.

    ``text`` holds the matchable characters (e.g. a greedy CTC decoding) and
    ``positions[i]`` is the position of ``text[i]`` in whatever units the
    caller wants boundaries expressed in -- CTC emission frames here.

    The n-gram index used by the search is built lazily on first use and
    cached on the instance. The TS original keys this cache off a ``WeakMap``
    so it does not outlive the document; a plain instance attribute does the
    same job in Python (the index dies with the document, no explicit
    invalidation needed since the object is otherwise immutable in practice).
    """

    text: str
    positions: np.ndarray
    _ngram_index: Optional[Dict[str, List[int]]] = field(default=None, repr=False)


@dataclass
class BoundaryMatch:
    """A located span of ``query`` inside a :class:`PositionedDocument`.

    ``start``/``end`` are document positions (CTC frames, already padded and
    clamped). ``evidence_range``/``matched_range`` are QUERY CHARACTER offset
    ranges (not document positions): ``matched_range`` is the region actually
    spanned by inlier n-grams, ``evidence_range`` is that region grown by a
    gap-derived margin. ``anchors`` are ``(query_offset, frame)`` pairs, kept
    monotonic in both coordinates.
    """

    start: int
    end: int
    confidence: float
    anchors: List[Tuple[int, int]]
    evidence_range: Tuple[int, int]
    matched_range: Tuple[int, int]


@dataclass
class ChapterSearchResult:
    """The outcome of searching one spine chapter's text in the document.

    ``start_char``/``end_char`` are the chapter's char range in ``full_text``
    (as given to :func:`search_chapters`), not the (possibly chunked) query's
    own range. ``anchors`` are ``(book_char_offset, frame)`` pairs -- query
    offsets translated back through ``build_query``'s offset map -- merged
    across every found chunk of a chunked chapter.
    """

    chapter_index: int
    start_char: int
    end_char: int
    found: bool
    start_frame: Optional[int]
    end_frame: Optional[int]
    confidence: Optional[float]
    anchors: List[Tuple[int, int]]


# --------------------------------------------------------------------------- #
# Greedy CTC decode
# --------------------------------------------------------------------------- #


def greedy_decode(
    log_probs: np.ndarray, blank_id: int, id_to_char: Dict[int, str]
) -> Tuple[str, np.ndarray]:
    """Greedy CTC decode: per-frame argmax, emitting a char on argmax change.

    Port of ``ctcGreedyDecode``. ``log_probs`` is ``[T, C]``. A character is
    emitted for a frame when its argmax token differs from the previous
    frame's argmax AND that token maps to a single-character entry in
    ``id_to_char`` -- this mirrors the TS building its ``charByToken`` map by
    excluding the blank id and any multi-character/separator token (e.g.
    ``"|"``) before decoding; skip blank and any token not present in
    ``id_to_char`` here the same way.

    Returns ``(text, frames)`` where ``frames[i]`` is the emission frame at
    which ``text[i]`` was first observed.
    """
    t_count = log_probs.shape[0] if log_probs.ndim == 2 else 0
    if t_count == 0:
        return "", np.array([], dtype=np.int64)

    return greedy_decode_argmax(np.argmax(log_probs, axis=1), blank_id, id_to_char)


def greedy_decode_argmax(
    argmaxes: np.ndarray, blank_id: int, id_to_char: Dict[int, str]
) -> Tuple[str, np.ndarray]:
    """``greedy_decode`` from per-frame argmax ids already computed by the caller.

    Lets a caller holding GPU emissions take the argmax on the device and copy
    only ``[T]`` ids to the host, instead of the whole ``[T, C]`` float array
    (about 460 MB for a 22-hour book).
    """
    if len(argmaxes) == 0:
        return "", np.array([], dtype=np.int64)

    change_mask = np.empty(len(argmaxes), dtype=bool)
    change_mask[0] = True
    if len(argmaxes) > 1:
        change_mask[1:] = argmaxes[1:] != argmaxes[:-1]

    lookup: Dict[int, str] = {
        token: char
        for token, char in id_to_char.items()
        if token != blank_id and len(char) == 1
    }

    chars: List[str] = []
    frames: List[int] = []
    for t in np.nonzero(change_mask)[0]:
        token = int(argmaxes[t])
        char = lookup.get(token)
        if char is not None:
            chars.append(char)
            frames.append(int(t))

    return "".join(chars), np.array(frames, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Query construction
# --------------------------------------------------------------------------- #


def build_query(full_text: str, start: int, end: int) -> Tuple[str, List[int]]:
    """Normalize ``full_text[start:end]`` into a CTC search query.

    Replicates the forced-aligner's MMS_FA character rule (``_MMS_CHARS_RE``
    in ``src/utils/forced_aligner.py``: lowercase, keep only ``[a-z']``,
    everything else dropped) but per character rather than per
    whitespace-delimited word, so every kept query character maps back to its
    own canonical offset in ``full_text`` and the query has no spaces
    (matching the CTC decoder's space-free vocabulary).

    Returns ``(query, book_offsets)`` where ``book_offsets[i]`` is the
    ``full_text`` char offset of ``query[i]``.
    """
    chars: List[str] = []
    offsets: List[int] = []
    for offset in range(start, end):
        lowered = full_text[offset].lower()
        if len(lowered) == 1 and _QUERY_KEEP_RE.match(lowered):
            chars.append(lowered)
            offsets.append(offset)
    return "".join(chars), offsets


# --------------------------------------------------------------------------- #
# Internal ported helpers (search.ts)
# --------------------------------------------------------------------------- #


def _median(values: List[float]) -> float:
    """``sorted[floor(len/2)]`` -- matches the TS helper (not a true median
    for even-length inputs, deliberately)."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _make_rng(seed: int = 0x2545F491) -> Callable[[], float]:
    """Deterministic LCG matching the TS ``random()`` closure in ``fitLine``.

    ``seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff``. Python ints
    are arbitrary precision, so ``(a * b) & 0xFFFFFFFF`` reproduces
    ``Math.imul``'s 32-bit wraparound multiply bit-for-bit: the low 32 bits of
    the exact product equal Math.imul's result regardless of sign
    interpretation, and only the low 31 bits of the whole expression survive
    the final ``& 0x7fffffff`` mask, so no explicit signed reinterpretation is
    needed at any step.
    """
    state = [seed]

    def _next() -> float:
        state[0] = (((state[0] * 1103515245) & 0xFFFFFFFF) + 12345) & 0x7FFFFFFF
        return state[0] / 0x80000000

    return _next


def _js_round(x: float) -> int:
    """``Math.round``: round-half-up (towards +Infinity), not Python's
    round-half-to-even."""
    return math.floor(x + 0.5)


def _get_ngram_index(document: PositionedDocument) -> Dict[str, List[int]]:
    """Every ``NGRAM_SIZE``-char n-gram of ``document.text`` -> document
    positions, cached on the document."""
    if document._ngram_index is not None:
        return document._ngram_index

    text = document.text
    positions = document.positions
    index: Dict[str, List[int]] = {}
    for i in range(len(text) - NGRAM_SIZE + 1):
        ngram = text[i : i + NGRAM_SIZE]
        position = int(positions[i])
        index.setdefault(ngram, []).append(position)

    document._ngram_index = index
    return index


def _collect_matches(
    query: str,
    document: PositionedDocument,
    excluded_ranges: List[Tuple[int, int]],
) -> Tuple[List[Match], Dict[str, List[int]]]:
    """Every query n-gram's document matches (outside ``excluded_ranges``),
    plus how many times each n-gram occurs in the query itself."""
    index = _get_ngram_index(document)

    query_ngram_offsets: Dict[str, List[int]] = {}
    matches: List[Match] = []
    for i in range(len(query) - NGRAM_SIZE + 1):
        q_ngram = query[i : i + NGRAM_SIZE]
        query_ngram_offsets.setdefault(q_ngram, []).append(i)

        positions = index.get(q_ngram)
        if not positions:
            continue

        for position in positions:
            if any(start <= position < end for start, end in excluded_ranges):
                continue
            matches.append(Match(offset=i, position=position))

    return matches, query_ngram_offsets


def _fit_line(
    matches: List[Match],
    pair_separation: float,
    min_slope: float,
    max_slope: float,
    inlier_tolerance: float,
) -> Optional[_LineFit]:
    """RANSAC over (offset, position) pairs, vectorized inlier counting.

    500 iterations picking two random matches via the deterministic LCG,
    fitting a line through them (rejecting pairs too close together or
    outside the slope range), and keeping the line with the most inliers
    within ``inlier_tolerance``. The winning slope/intercept are then
    re-estimated from the inlier set's medians.
    """
    if not matches:
        return None

    n = len(matches)
    offsets = np.array([m.offset for m in matches], dtype=np.float64)
    positions = np.array([m.position for m in matches], dtype=np.float64)

    random = _make_rng()

    best_slope = 0.0
    best_intercept = 0.0
    best_count = -1

    for _ in range(RANSAC_ITERATIONS):
        ia = min(int(random() * n), n - 1)
        ib = min(int(random() * n), n - 1)
        a = matches[ia]
        b = matches[ib]

        offset_delta = b.offset - a.offset
        if abs(offset_delta) < pair_separation:
            continue

        slope = (b.position - a.position) / offset_delta
        if slope < min_slope or slope > max_slope:
            continue

        intercept = a.position - slope * a.offset

        predicted = intercept + slope * offsets
        count = int(np.count_nonzero(np.abs(positions - predicted) <= inlier_tolerance))

        if count > best_count:
            best_slope, best_intercept, best_count = slope, intercept, count

    if best_count < MIN_INLIERS:
        return None

    predicted_all = best_intercept + best_slope * offsets
    inlier_mask = np.abs(positions - predicted_all) <= inlier_tolerance
    inliers = [m for m, keep in zip(matches, inlier_mask) if keep]

    sorted_inliers = sorted(inliers, key=lambda m: m.offset)

    stride = max(len(sorted_inliers) // 2, 1)
    slopes: List[float] = []
    for i in range(len(sorted_inliers) - stride):
        lo = sorted_inliers[i]
        hi = sorted_inliers[i + stride]
        if hi.offset - lo.offset < pair_separation:
            continue
        slopes.append((hi.position - lo.position) / (hi.offset - lo.offset))

    slope = _median(slopes) if slopes else best_slope
    intercept = (
        _median([m.position - slope * m.offset for m in sorted_inliers])
        if sorted_inliers
        else best_intercept
    )

    return _LineFit(slope=slope, intercept=intercept, inliers=sorted_inliers)


def _bigram_similarity(a: str, b: str) -> float:
    """Multiset bigram overlap between two strings, normalized by the longer
    string's bigram count."""
    if len(a) < 2 or len(b) < 2:
        return 0.0

    bigrams: Dict[str, int] = {}
    for i in range(len(a) - 1):
        bigram = a[i : i + 2]
        bigrams[bigram] = bigrams.get(bigram, 0) + 1

    shared = 0
    for i in range(len(b) - 1):
        bigram = b[i : i + 2]
        remaining = bigrams.get(bigram, 0)
        if remaining > 0:
            shared += 1
            bigrams[bigram] = remaining - 1

    return shared / max(len(a) - 1, len(b) - 1)


def _position_to_document_index(document: PositionedDocument, position: float) -> int:
    """Lower-bound binary search: the first document index whose position is
    ``>= position``, clamped into ``[0, len(positions) - 1]`` (matches the
    TS's index-bounded binary search exactly, including its behaviour when
    ``position`` exceeds every element)."""
    positions = document.positions
    low, high = 0, len(positions) - 1
    while low < high:
        mid = (low + high) // 2
        if positions[mid] < position:
            low = mid + 1
        else:
            high = mid
    return low


def _context_agreement(match: Match, query: str, document: PositionedDocument) -> float:
    """Bigram similarity of the text surrounding ``match`` on the left and
    right, in the query vs. the document; the better-agreeing side wins."""
    index = _position_to_document_index(document, match.position)

    query_left = query[max(0, match.offset - ANCHOR_CONTEXT) : match.offset]
    document_left = document.text[max(0, index - ANCHOR_CONTEXT) : index]

    query_right = query[match.offset + NGRAM_SIZE : match.offset + NGRAM_SIZE + ANCHOR_CONTEXT]
    document_right = document.text[index + NGRAM_SIZE : index + NGRAM_SIZE + ANCHOR_CONTEXT]

    return max(
        _bigram_similarity(query_left, document_left),
        _bigram_similarity(query_right, document_right),
    )


def _select_anchors(
    query: str,
    inliers: List[Match],
    query_ngram_offsets: Dict[str, List[int]],
    document: PositionedDocument,
    slope: float,
    intercept: float,
    spacing: int = ANCHOR_SPACING,
) -> List[Match]:
    """One anchor per ~``spacing`` query chars (``ANCHOR_SPACING`` in the TS), where the n-gram is
    unique in both the query and the document and (after calibration) its
    context agrees well enough; kept monotonic in both coordinates."""
    document_ngram_index = _get_ngram_index(document)

    agreements: Dict[Match, float] = {}

    def agreement(match: Match) -> float:
        score = agreements.get(match)
        if score is None:
            score = _context_agreement(match, query, document)
            agreements[match] = score
        return score

    def select(min_agreement: Optional[float]) -> List[Match]:
        anchors: List[Match] = []
        for i in range(0, len(query), spacing):
            window_start = max(0, i - spacing // 2)
            window_end = min(len(query), i + spacing // 2)

            candidates = sorted(
                (m for m in inliers if window_start <= m.offset < window_end),
                key=lambda m: abs(m.offset - i),
            )

            for c in candidates:
                gram = query[c.offset : c.offset + NGRAM_SIZE]

                if len(document_ngram_index.get(gram, [])) != 1:
                    continue
                if len(query_ngram_offsets.get(gram, [])) != 1:
                    continue
                if min_agreement is not None and agreement(c) < min_agreement:
                    continue

                anchors.append(c)
                break

        return anchors

    anchors = select(None)
    if len(anchors) >= ANCHOR_CALIBRATION_MINIMUM:
        threshold = max(
            ANCHOR_AGREEMENT_FRACTION * _median([agreement(a) for a in anchors]),
            ANCHOR_AGREEMENT_FLOOR,
        )
        anchors = select(threshold)

    monotonic_anchors: List[Match] = []
    for a in anchors:
        while True:
            if not monotonic_anchors:
                monotonic_anchors.append(a)
                break

            b = monotonic_anchors[-1]
            if a.position - b.position >= a.offset - b.offset:
                monotonic_anchors.append(a)
                break

            a_dist = abs(a.position - (intercept + slope * a.offset))
            b_dist = abs(b.position - (intercept + slope * b.offset))
            if a_dist > b_dist:
                break
            monotonic_anchors.pop()

    return monotonic_anchors


def _fit_local_line(points: List[Match], fallback_slope: float) -> Tuple[float, float]:
    stride = max(len(points) // 2, 1)
    slopes: List[float] = []
    for i in range(0, len(points) - stride, stride):
        lo = points[i]
        hi = points[i + stride]
        if hi.offset - lo.offset < EDGE_EXTENSION_PAIR_SEPARATION:
            continue
        slopes.append((hi.position - lo.position) / (hi.offset - lo.offset))

    slope = _median(slopes) if slopes else fallback_slope
    intercept = _median([m.position - slope * m.offset for m in points])
    return slope, intercept


def _extend_edge(
    sorted_matches: List[Match],
    sorted_inliers: List[Match],
    global_slope: float,
    inlier_tolerance: float,
) -> List[Match]:
    """Walk forward from the last inlier, accepting further matches that fit
    a locally re-fit line, up to a max gap. Both argument lists (and the
    result) are in ascending-offset order."""
    if not sorted_inliers:
        return []
    last_inlier = sorted_inliers[-1]

    window = list(sorted_inliers[-EDGE_EXTENSION_WINDOW:])
    accepted: List[Match] = []
    last_accepted_offset = last_inlier.offset
    slope, intercept = _fit_local_line(window, global_slope)

    for match in sorted_matches:
        if match.offset <= last_inlier.offset:
            continue
        if match.offset - last_accepted_offset > EDGE_EXTENSION_MAX_GAP:
            break

        predicted = intercept + slope * match.offset
        if abs(match.position - predicted) > inlier_tolerance:
            continue

        accepted.append(match)
        window.append(match)
        if len(window) > EDGE_EXTENSION_WINDOW:
            window.pop(0)
        last_accepted_offset = match.offset
        slope, intercept = _fit_local_line(window, global_slope)

    return accepted


def _extend_inlier_edges(
    matches: List[Match],
    sorted_inliers: List[Match],
    slope: float,
    inlier_tolerance: float,
) -> List[Match]:
    """Extend the inlier run in both directions by walking the full match
    list outward from each edge (the head extension runs on a
    offset/position-negated mirror so the same forward-walking logic applies)."""
    by_offset = sorted(matches, key=lambda m: m.offset)

    tail = _extend_edge(by_offset, sorted_inliers, slope, inlier_tolerance)

    def mirror(m: Match) -> Match:
        return Match(offset=-m.offset, position=-m.position)

    mirrored_matches = [mirror(m) for m in reversed(by_offset)]
    mirrored_inliers = [mirror(m) for m in reversed(sorted_inliers)]
    head = [
        mirror(m)
        for m in reversed(_extend_edge(mirrored_matches, mirrored_inliers, slope, inlier_tolerance))
    ]

    return head + sorted_inliers + tail


def compute_evidence_range(
    offsets: List[int], query_length: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Port of ``computeEvidenceRange``.

    ``offsets`` is an ascending, de-duplicated list of inlier query offsets.
    Returns ``(evidence_range, matched_range)``, both ``(start, end)`` QUERY
    CHARACTER offset pairs: ``matched_range`` spans the offsets actually
    covered (after trimming edge clusters too small to trust), and
    ``evidence_range`` grows that span by a gap-derived margin (or to the
    query's edge, if the margin would reach past it).
    """
    if len(offsets) < 2:
        return (0, query_length), (0, query_length)

    gaps = [offsets[i] - offsets[i - 1] for i in range(1, len(offsets))]
    sorted_gaps = sorted(gaps)
    p99_index = min(len(sorted_gaps) - 1, int(len(sorted_gaps) * EVIDENCE_GAP_PERCENTILE))
    p99 = sorted_gaps[p99_index]

    threshold = max(EVIDENCE_GAP_MULTIPLIER * p99, EVIDENCE_GAP_FLOOR)

    clusters: List[Tuple[int, int]] = []
    cluster_start = 0
    for i, gap in enumerate(gaps, start=1):
        if gap <= threshold:
            continue
        clusters.append((cluster_start, i - 1))
        cluster_start = i
    clusters.append((cluster_start, len(offsets) - 1))

    first_cluster = 0
    last_cluster = len(clusters) - 1
    while (
        first_cluster < last_cluster
        and clusters[first_cluster][1] - clusters[first_cluster][0] + 1 < MIN_EDGE_CLUSTER_OFFSETS
    ):
        first_cluster += 1
    while (
        last_cluster > first_cluster
        and clusters[last_cluster][1] - clusters[last_cluster][0] + 1 < MIN_EDGE_CLUSTER_OFFSETS
    ):
        last_cluster -= 1

    first = offsets[clusters[first_cluster][0]]
    last = offsets[clusters[last_cluster][1]] + NGRAM_SIZE

    matched_range = (first, last)
    evidence_range = (
        first - threshold if first > threshold else 0,
        last + threshold if query_length - last > threshold else query_length,
    )
    return evidence_range, matched_range


def _find_boundaries_in_document(
    query: str,
    document: PositionedDocument,
    excluded_ranges: List[Tuple[int, int]],
    min_slope: float,
    max_slope: float,
    inlier_tolerance: float,
    anchor_spacing: int = ANCHOR_SPACING,
) -> Optional[BoundaryMatch]:
    """Port of ``findBoundariesInDocument``: the full fit/extend/score
    pipeline, unclamped (see :func:`find_boundaries` for the frame clamp)."""
    gram_count = max(len(query) - NGRAM_SIZE + 1, 0)
    pair_separation = min(max(gram_count // 4, 20), MAX_PAIR_SEPARATION_REQUIREMENT)

    matches, query_ngram_offsets = _collect_matches(query, document, excluded_ranges)
    if len(matches) < MIN_INLIERS:
        return None

    line = _fit_line(matches, pair_separation, min_slope, max_slope, inlier_tolerance)
    if line is None:
        return None

    slope, intercept, inliers = line.slope, line.intercept, line.inliers

    by_offset = _extend_inlier_edges(
        matches, sorted(inliers, key=lambda m: m.offset), slope, inlier_tolerance
    )
    inlier_offsets = list(dict.fromkeys(m.offset for m in by_offset))

    evidence_range, matched_range = compute_evidence_range(inlier_offsets, len(query))

    head = by_offset[:REFINE_COUNT]
    tail = by_offset[-REFINE_COUNT:]

    extrapolated_start = min(m.position - slope * m.offset for m in head)
    extrapolated_end = max(m.position + slope * (len(query) - m.offset) for m in tail)

    start_pad = max(BOUNDARY_PAD_FRAMES, 0.25 * (head[0].position - extrapolated_start))
    start = extrapolated_start - start_pad

    end_pad = max(BOUNDARY_PAD_FRAMES, 0.25 * (extrapolated_end - tail[-1].position))
    end = extrapolated_end + end_pad

    if end <= start:
        return None

    matched_grams = len({m.offset for m in matches})
    inlier_grams = len(inlier_offsets)
    confidence = inlier_grams / matched_grams
    if confidence < MIN_CONFIDENCE:
        return None

    first = by_offset[0]
    last = by_offset[-1]
    coverage = (last.offset - first.offset + NGRAM_SIZE) / len(query)
    if coverage < MIN_COVERAGE:
        return None

    anchors = _select_anchors(query, inliers, query_ngram_offsets, document, slope, intercept,
                              spacing=anchor_spacing)

    return BoundaryMatch(
        start=_js_round(start),
        end=_js_round(end),
        confidence=confidence,
        anchors=[(m.offset, m.position) for m in anchors],
        evidence_range=evidence_range,
        matched_range=matched_range,
    )


# --------------------------------------------------------------------------- #
# Public search API
# --------------------------------------------------------------------------- #


def find_boundaries(
    query: str,
    document: PositionedDocument,
    excluded_ranges: List[Tuple[int, int]],
    *,
    min_slope: float = 2,
    max_slope: float = 15,
    inlier_tolerance: Optional[float] = None,
    num_frames: Optional[int] = None,
    anchor_spacing: int = ANCHOR_SPACING,
) -> Optional[BoundaryMatch]:
    """Locate ``query`` inside ``document``, clamped to valid frames.

    ``anchor_spacing`` (query chars between anchors) defaults to Storyteller's
    ``ANCHOR_SPACING``; a caller that windows a chunked aligner on the anchors
    can ask for denser ones.

    Port of ``findCtcBoundaries``'s wrapping of ``findBoundariesInDocument``.
    ``excluded_ranges`` are ``[start, end)`` document-position ranges (e.g.
    frames already claimed by a previously found chapter) whose matches are
    dropped before fitting. When ``inlier_tolerance`` is not given it is
    derived exactly as ``findCtcBoundaries`` does: ``5000`` for queries over
    15,000 chars, else ``2500``. ``start`` is clamped to ``>= 0``; ``end`` is
    clamped to ``<= num_frames - 1`` when ``num_frames`` is given.
    """
    if inlier_tolerance is None:
        inlier_tolerance = 5000 if len(query) > 15000 else 2500

    result = _find_boundaries_in_document(
        query, document, excluded_ranges, min_slope, max_slope, inlier_tolerance,
        anchor_spacing=anchor_spacing,
    )
    if result is None:
        return None

    start = max(result.start, 0)
    end = result.end
    if num_frames is not None:
        end = min(end, num_frames - 1)

    return replace(result, start=start, end=end)


def _narrow_to_available_boundary(
    claimed: List[Tuple[int, int]], boundary_start: int, boundary_end: int
) -> Tuple[int, int]:
    """Port of ``narrowToAvailableBoundary``: shrink ``[boundary_start,
    boundary_end]`` to the largest stretch not already covered by a
    ``claimed`` range. Returns the boundary unchanged if it does not
    intersect any unclaimed stretch (the TS's own fallback)."""
    merged: List[List[int]] = []
    for start, end in sorted(claimed, key=lambda p: p[0]):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    available: List[float] = [-1]
    for start, end in merged:
        available.append(start)
        available.append(end)
    available.append(math.inf)

    within_boundary: List[Tuple[float, float]] = []
    for i in range(0, len(available) - 1, 2):
        seg_start, seg_end = available[i], available[i + 1]
        if (boundary_start <= seg_start <= boundary_end) or (
            boundary_start <= seg_end <= boundary_end
        ):
            narrowed_start = max(boundary_start, seg_start + 1)
            narrowed_end = min(boundary_end, seg_end - 1)
            if narrowed_start <= narrowed_end:
                within_boundary.append((narrowed_start, narrowed_end))

    if not within_boundary:
        return boundary_start, boundary_end

    largest = max(within_boundary, key=lambda p: p[1] - p[0])
    return int(largest[0]), int(largest[1])


def search_chapters(
    document: PositionedDocument,
    full_text: str,
    chapters: List[Tuple[int, int]],
    num_frames: int,
    anchor_spacing: int = ANCHOR_SPACING,
) -> List[ChapterSearchResult]:
    """Locate each spine chapter's text inside ``document``, in spine order.

    ``chapters`` are ``(start_char, end_char)`` pairs into ``full_text``, in
    spine order. Port of the search-and-exclude loop in ``Aligner.ts``'s
    ``processSpineItem``/``getMatchedBoundaries``: frame ranges claimed by a
    previously *found* chapter or chunk are excluded from every later search
    (via both the n-gram match filter and ``narrowToAvailableBoundary``), so
    a duplicated passage cannot be matched twice.

    A chapter whose normalized query is shorter than ``2 * NGRAM_SIZE`` is
    reported not found without searching. A chapter whose query exceeds
    ``MAX_SEARCH_LENGTH`` (40,000) chars is split into evenly sized chunks and
    each is searched independently, in its own 0-based offset space, with
    exclusions carried over between chunks the same way they carry over
    between chapters. **Simplification from the TS**: ``Aligner.ts`` chunks a
    chapter at sentence boundaries; our query has no whitespace at all (it is
    normalized to bare ``[a-z']`` characters), so there is no sentence
    structure to chunk on -- this splits by raw query length instead, at
    roughly even chunk sizes (mirroring the TS's own
    ``Math.ceil(length / count)`` chunk-length derivation), which produces
    chapter boundaries at arbitrary decoded-character positions rather than
    sentence ends. A chapter is found if at least one of its chunks is found;
    ``start_frame``/``end_frame`` span the found chunks' narrowed ranges, and
    ``confidence`` is the minimum across found chunks (another simplification
    -- the TS has no single per-chapter confidence for a chunked chapter; the
    minimum is the conservative choice for gating downstream use).
    """
    claimed_ranges: List[Tuple[int, int]] = []
    results: List[ChapterSearchResult] = []

    for chapter_index, (start_char, end_char) in enumerate(chapters):
        query, book_offsets = build_query(full_text, start_char, end_char)

        if len(query) < 2 * NGRAM_SIZE:
            results.append(
                ChapterSearchResult(
                    chapter_index=chapter_index,
                    start_char=start_char,
                    end_char=end_char,
                    found=False,
                    start_frame=None,
                    end_frame=None,
                    confidence=None,
                    anchors=[],
                )
            )
            continue

        chunk_count = max(1, math.ceil(len(query) / MAX_SEARCH_LENGTH))
        chunk_length = math.ceil(len(query) / chunk_count)

        chunk_frames: List[Tuple[int, int]] = []
        chunk_confidences: List[float] = []
        chapter_anchors: List[Tuple[int, int]] = []

        for chunk_start in range(0, len(query), chunk_length):
            chunk_end = min(chunk_start + chunk_length, len(query))
            chunk_query = query[chunk_start:chunk_end]
            chunk_book_offsets = book_offsets[chunk_start:chunk_end]

            boundary = find_boundaries(
                chunk_query,
                document,
                claimed_ranges,
                min_slope=2,
                max_slope=15,
                num_frames=num_frames,
                anchor_spacing=anchor_spacing,
            )
            if boundary is None:
                continue

            narrowed_start, narrowed_end = _narrow_to_available_boundary(
                claimed_ranges, boundary.start, boundary.end
            )
            if narrowed_start == narrowed_end:
                continue

            claimed_ranges.append((narrowed_start, narrowed_end))
            chunk_frames.append((narrowed_start, narrowed_end))
            chunk_confidences.append(boundary.confidence)
            chapter_anchors.extend(
                (chunk_book_offsets[q_offset], frame) for q_offset, frame in boundary.anchors
            )

        if not chunk_frames:
            results.append(
                ChapterSearchResult(
                    chapter_index=chapter_index,
                    start_char=start_char,
                    end_char=end_char,
                    found=False,
                    start_frame=None,
                    end_frame=None,
                    confidence=None,
                    anchors=[],
                )
            )
            continue

        results.append(
            ChapterSearchResult(
                chapter_index=chapter_index,
                start_char=start_char,
                end_char=end_char,
                found=True,
                start_frame=min(s for s, _ in chunk_frames),
                end_frame=max(e for _, e in chunk_frames),
                confidence=min(chunk_confidences),
                anchors=chapter_anchors,
            )
        )

    return results
