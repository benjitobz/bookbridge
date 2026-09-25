"""The background alignment job hands a finished book to read-along generation.

Reported bug: on the Storyteller-free path (Add Book with "Also generate a
read-along EPUB", then Match All), the request was consumed only by the
Storyteller forge completion hook, so a book aligned by the normal background
job never got its read-along. `SyncManager._run_background_job` now calls the
injected dispatcher (`ForgeService.generate_readalong_if_requested`, which
consumes the request exactly once) after a completed alignment.
"""
from unittest.mock import MagicMock

from src.db.models import Book
from tests.test_storyteller_priority_flow import _build_manager


def _pending_book(tmp_path) -> Book:
    return Book(
        abs_id="bookorbit:5143",
        abs_title="Readalong Book",
        ebook_filename="book.epub",
        kosync_doc_id="hash-ra-1",
        status="pending",
        duration=12.0,
    )


def test_completed_alignment_dispatches_the_readalong_request(tmp_path):
    manager, _db, _abs, _transcriber, _alignment = _build_manager(tmp_path)
    dispatcher = MagicMock()
    manager.readalong_intent_dispatcher = dispatcher

    manager._run_background_job(_pending_book(tmp_path))

    dispatcher.assert_called_once()
    assert dispatcher.call_args.args[0].abs_id == "bookorbit:5143"
    assert dispatcher.call_args.args[0].status == "active"


def test_failed_alignment_does_not_dispatch(tmp_path):
    manager, _db, _abs, _transcriber, alignment_service = _build_manager(tmp_path)
    alignment_service.align_and_store.return_value = False
    alignment_service.align_storyteller_and_store.return_value = False
    dispatcher = MagicMock()
    manager.readalong_intent_dispatcher = dispatcher

    manager._run_background_job(_pending_book(tmp_path))

    dispatcher.assert_not_called()


def test_a_dispatcher_error_does_not_fail_the_alignment_job(tmp_path, caplog):
    manager, db, _abs, _transcriber, _alignment = _build_manager(tmp_path)
    manager.readalong_intent_dispatcher = MagicMock(side_effect=RuntimeError("boom"))
    book = _pending_book(tmp_path)

    manager._run_background_job(book)

    assert book.status == "active"
    assert "post-alignment dispatch failed" in caplog.text
    failed_marks = [
        c for c in db.update_latest_job.call_args_list
        if (c.kwargs.get("last_error") or "")
    ]
    assert failed_marks == []


def test_no_dispatcher_configured_is_a_noop(tmp_path):
    manager, _db, _abs, _transcriber, _alignment = _build_manager(tmp_path)
    manager.readalong_intent_dispatcher = None
    book = _pending_book(tmp_path)

    manager._run_background_job(book)

    assert book.status == "active"
