""""What's new after upgrade" banner: decides whether a logged-in user should
be shown the post-upgrade banner/page, and parses ``RELEASE_NOTES.md`` into
the pieces the banner and the ``/whats-new`` page need.

This module is intentionally free of Flask/DB imports: ``should_show_banner``
is a pure function over primitives so it is trivial to unit test, and the
notes loader only touches the filesystem. Callers (``src/web_server.py``) own
reading ``APP_VERSION``, the logged-in user, and the per-user stored value,
and own persisting the "seen" version via ``DatabaseService``.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import markdown

from src.utils.time_utils import utcnow

logger = logging.getLogger(__name__)

# Baked into the image next to the app code (see Dockerfile); falls back to
# the repo-root copy for local/dev runs where /app doesn't exist.
_PRIMARY_NOTES_PATH = Path("/app/RELEASE_NOTES.md")
_FALLBACK_NOTES_PATH = Path(__file__).resolve().parents[2] / "RELEASE_NOTES.md"

_ACTION_REQUIRED_HEADING = "Action Required"

# Recorded once, at import time: this process's start time. Used to
# distinguish "existing user who just upgraded" (created before the process
# that's now running started) from "brand-new account" (created by this same
# process, e.g. during initial setup) when no seen-version has been stored
# yet for the user.
PROCESS_STARTED_AT: datetime = utcnow()


@dataclass(frozen=True)
class ParsedReleaseNotes:
    """The pieces of RELEASE_NOTES.md the banner and /whats-new page need."""
    action_required_items: List[str] = field(default_factory=list)
    html: str = ""


_notes_cache: Optional[ParsedReleaseNotes] = None


def _find_notes_path() -> Optional[Path]:
    """Locate RELEASE_NOTES.md: the baked-in image copy, else the repo-root
    copy for local/dev runs. None if neither exists."""
    if _PRIMARY_NOTES_PATH.exists():
        return _PRIMARY_NOTES_PATH
    if _FALLBACK_NOTES_PATH.exists():
        return _FALLBACK_NOTES_PATH
    return None


def _split_sections(text: str) -> "dict[Optional[str], str]":
    """Split a release-notes markdown document into ``{heading: body}`` by
    level-2 (``##``) headings. The text before the first ``##`` (the intro
    paragraph and the top-level ``#`` title) is kept under the ``None`` key.
    Section order matches the order headings appear in the source text."""
    sections: "dict[Optional[str], str]" = {}
    current_key: Optional[str] = None
    buf: List[str] = []
    for line in text.splitlines():
        match = re.match(r'^##\s+(.+?)\s*$', line)
        if match:
            sections[current_key] = "\n".join(buf).strip()
            current_key = match.group(1).strip()
            buf = []
        else:
            buf.append(line)
    sections[current_key] = "\n".join(buf).strip()
    return sections


def _extract_bullets(section_text: str) -> List[str]:
    """Extract top-level ``- `` bullets from a markdown section as plain
    text, joining wrapped continuation lines and stripping ``**bold**``
    markers (the banner shows this text inline, not as markdown)."""
    bullets: List[str] = []
    current: Optional[str] = None
    for line in section_text.splitlines():
        if re.match(r'^-\s+', line):
            if current is not None:
                bullets.append(current.strip())
            current = re.sub(r'^-\s+', '', line)
        elif current is not None and line.strip():
            current += " " + line.strip()
    if current is not None:
        bullets.append(current.strip())
    return [re.sub(r'\*\*(.+?)\*\*', r'\1', b) for b in bullets]


def _render_html(sections: "dict[Optional[str], str]") -> str:
    """Reassemble the sections into markdown with Action Required moved
    first (right after the intro), then render to HTML."""
    parts: List[str] = []
    intro = sections.get(None, "")
    if intro:
        parts.append(intro)
    if _ACTION_REQUIRED_HEADING in sections:
        parts.append(f"## {_ACTION_REQUIRED_HEADING}\n\n{sections[_ACTION_REQUIRED_HEADING]}")
    for heading, body in sections.items():
        if heading in (None, _ACTION_REQUIRED_HEADING):
            continue
        parts.append(f"## {heading}\n\n{body}")
    full_markdown = "\n\n".join(p for p in parts if p)
    return markdown.markdown(full_markdown, extensions=["fenced_code"])


def load_release_notes(force_reload: bool = False) -> Optional[ParsedReleaseNotes]:
    """Load, parse and cache RELEASE_NOTES.md. Returns ``None`` (never
    raises) when the file is missing or unreadable; the file only changes
    with the image, so the parsed result is cached in-process."""
    global _notes_cache
    if _notes_cache is not None and not force_reload:
        return _notes_cache

    path = _find_notes_path()
    if path is None:
        logger.debug(
            f"RELEASE_NOTES.md not found at {_PRIMARY_NOTES_PATH} or {_FALLBACK_NOTES_PATH}"
        )
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug(f"Could not read release notes at {path}: {e}", exc_info=True)
        return None

    sections = _split_sections(text)
    parsed = ParsedReleaseNotes(
        action_required_items=_extract_bullets(sections.get(_ACTION_REQUIRED_HEADING, "")),
        html=_render_html(sections),
    )
    _notes_cache = parsed
    return parsed


def get_action_required_items() -> List[str]:
    """Plain-text Action Required bullets for the banner, or ``[]`` if the
    notes are missing or have no such section."""
    notes = load_release_notes()
    return notes.action_required_items if notes else []


def get_release_notes_html() -> str:
    """Rendered HTML for the /whats-new page, or ``""`` if the notes are
    unavailable."""
    notes = load_release_notes()
    return notes.html if notes else ""


def should_show_banner(
    app_version: str,
    stored_value: Optional[str],
    user_created_at: Optional[datetime],
    process_started_at: datetime,
) -> str:
    """Decide what the caller should do for this user, given the version and
    per-user stored state. Pure function; the caller performs any DB write.

    Returns one of:
      - ``"hide"``: no banner (dev build, or the stored version already
        matches the running version).
      - ``"show"``: display the banner.
      - ``"store_current"``: no banner this time, but the caller should
        store ``app_version`` as seen for this user (first-run problem: a
        brand-new account created by this same process has nothing to
        announce "what's new" about).
    """
    if not app_version or app_version.startswith("dev"):
        return "hide"

    if stored_value is not None:
        return "show" if stored_value != app_version else "hide"

    # No stored value yet. An existing user (created before this process
    # started) who has never had a version recorded just upgraded and should
    # see the banner. A user created by this same process (fresh install /
    # brand-new account) has nothing to announce -- silently baseline them.
    if user_created_at is not None and user_created_at < process_started_at:
        return "show"
    return "store_current"
