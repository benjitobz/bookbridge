"""Repair BookFusion links that contain Calibre/reader-facing ids.

BookFusion's Calibre upload API returns the id embedded in ``read_url``. Progress
endpoints use the distinct ``id`` returned by the User API search endpoint. This
script resolves each configured stale link through that User API and, after a
successful reading-position check, replaces it using ``DatabaseService``.

Run inside the BookBridge container (or another environment with the same data
directory and configuration). It is dry-run by default:

    python scripts/repair_bookfusion_user_api_ids.py --user-id 1 --data-dir /data
    python scripts/repair_bookfusion_user_api_ids.py --user-id 1 --data-dir /data --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bookfusion_client import BookFusionClient  # noqa: E402
from src.db.migration_utils import get_database_service  # noqa: E402

TARGET_ABS_IDS = (
    "ebook-16bc9e5360c6ac74",
    "ebook-d3663a2ef2225f05",
)


def _reader_id(item: dict) -> str | None:
    match = re.search(r"/books/(\d+)(?:[-/?#]|$)", str(item.get("read_url") or ""))
    return match.group(1) if match else None


def _first_author(item: dict) -> str:
    authors = item.get("authors") or item.get("author") or ""
    if isinstance(authors, list):
        first = authors[0] if authors else ""
        return str(first.get("name") or "").strip() if isinstance(first, dict) else str(first or "").strip()
    if isinstance(authors, dict):
        return str(authors.get("name") or "").strip()
    return str(authors).strip()


def _resolve_user_api_id(client: BookFusionClient, title: str, author: str, old_id: str) -> str | None:
    """Find the exact User-API record whose reader URL contains ``old_id``."""
    if not title or not author:
        return None
    results = client.search_books(page=1, per_page=20, q=title) or []
    for item in results:
        if (
            str(item.get("title") or "").strip().casefold() != title.casefold()
            or _first_author(item).casefold() != author.casefold()
            or _reader_id(item) != old_id
        ):
            continue
        candidate_id = str(item.get("id") or "").strip()
        if candidate_id:
            return candidate_id
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True, help="BookBridge user owning the BookFusion links")
    parser.add_argument("--data-dir", default="/data", help="Directory containing BookBridge database.db (default: /data)")
    parser.add_argument("--apply", action="store_true", help="Persist the repaired links; otherwise only report them")
    args = parser.parse_args(argv)

    db = get_database_service(args.data_dir)
    creds = db.get_user_credentials(args.user_id) or {}
    client = BookFusionClient(credentials=creds, database_service=db, user_id=args.user_id)
    if not client.is_configured():
        print("BookFusion is not configured for that user; no changes made.", file=sys.stderr)
        return 2

    unresolved = False
    for abs_id in TARGET_ABS_IDS:
        link = db.get_user_bookfusion_link(args.user_id, abs_id)
        if not link:
            print(f"{abs_id}: no BookFusion link; skipped")
            continue
        old_id = str(link["bookfusion_id"])
        title = str(link.get("title") or "").strip()
        author = str(link.get("author") or "").strip()
        new_id = _resolve_user_api_id(client, title, author, old_id)
        if not new_id:
            print(f"{abs_id}: could not resolve a User API id for {old_id}; skipped", file=sys.stderr)
            unresolved = True
            continue
        if not isinstance(client.get_reading_position(new_id), dict):
            print(f"{abs_id}: User API id {new_id} did not return a reading position; skipped", file=sys.stderr)
            unresolved = True
            continue
        print(f"{abs_id}: {old_id} -> {new_id}")
        if args.apply:
            db.set_user_bookfusion_link(
                args.user_id,
                abs_id,
                new_id,
                title=link.get("title"),
                author=link.get("author"),
            )

    if not args.apply:
        print("Dry run only. Re-run with --apply to persist the resolved mappings.")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
