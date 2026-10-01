"""The job queue must not run on the scheduler thread.

A long audiobook holds check_pending_jobs for hours; on the scheduler thread
that stalls every other scheduled job until the queue drains.
"""
import threading
import time

import pytest


@pytest.fixture
def web_server():
    import src.web_server as ws
    ws._pending_jobs_thread = None
    yield ws
    thread = ws._pending_jobs_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=5)
    ws._pending_jobs_thread = None


def _wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_queue_runs_off_the_calling_thread(web_server, monkeypatch):
    ran_on = {}
    started = threading.Event()

    def fake_check():
        ran_on["thread"] = threading.current_thread().name
        started.set()

    monkeypatch.setattr(web_server.manager, "check_pending_jobs", fake_check)

    assert web_server._check_pending_jobs_async() is True
    assert started.wait(timeout=5)
    assert ran_on["thread"] != threading.current_thread().name
    assert ran_on["thread"] == "pending-jobs"


def test_the_scheduler_is_not_blocked_by_a_long_job(web_server, monkeypatch):
    release = threading.Event()
    running = threading.Event()

    def slow_check():
        running.set()
        release.wait(timeout=10)

    monkeypatch.setattr(web_server.manager, "check_pending_jobs", slow_check)

    started = time.monotonic()
    assert web_server._check_pending_jobs_async() is True
    elapsed = time.monotonic() - started

    assert running.wait(timeout=5)
    assert elapsed < 1.0
    release.set()


def test_a_tick_while_the_worker_is_busy_is_a_no_op(web_server, monkeypatch):
    release = threading.Event()
    calls = []

    def slow_check():
        calls.append(1)
        release.wait(timeout=10)

    monkeypatch.setattr(web_server.manager, "check_pending_jobs", slow_check)

    assert web_server._check_pending_jobs_async() is True
    assert _wait(lambda: calls)
    first = web_server._pending_jobs_thread

    assert web_server._check_pending_jobs_async() is False
    assert web_server._pending_jobs_thread is first
    assert len(calls) == 1

    release.set()
    first.join(timeout=5)


def test_a_later_tick_starts_a_fresh_worker(web_server, monkeypatch):
    calls = []
    monkeypatch.setattr(web_server.manager, "check_pending_jobs", lambda: calls.append(1))

    assert web_server._check_pending_jobs_async() is True
    web_server._pending_jobs_thread.join(timeout=5)
    assert web_server._check_pending_jobs_async() is True
    web_server._pending_jobs_thread.join(timeout=5)

    assert _wait(lambda: len(calls) == 2)


def test_a_failing_job_does_not_wedge_the_worker(web_server, monkeypatch):
    def boom():
        raise RuntimeError("job exploded")

    monkeypatch.setattr(web_server.manager, "check_pending_jobs", boom)

    assert web_server._check_pending_jobs_async() is True
    web_server._pending_jobs_thread.join(timeout=5)

    monkeypatch.setattr(web_server.manager, "check_pending_jobs", lambda: None)
    assert web_server._check_pending_jobs_async() is True
