"""
Issue #447 — Storyteller progress yanked back to the ABS position every cycle.

Reporter (BookBridge v7.7.0 / main, Storyteller 2.14.21, ABS 2.36.1), log lines
verbatim (book id shortened):

    16:04:08 🔄 '2106c935-…' 'A Dance with Dragons (2011)' ABS leads at 44.0972% (normalized: 77666.2s, source=abs)
    16:04:08 ✅ Storyteller API: 19120e99... -> 43.7% (TS: 1790193848517)
      [the user reads on in the Storyteller app]
    16:07:09 📊 Storyteller: 43.6651% -> 43.2100%
    16:07:09 🪞 '2106c935-…' 'A Dance with Dragons (2011)' Ignoring 'Storyteller' delta (43.21%): it holds BookBridge's own write-back, not user movement
    16:07:09 🪞 '2106c935-…' 'A Dance with Dragons (2011)' Excluding own write-back candidate(s) ['ABSEbook', 'KoSync', 'Storyteller'] from leader selection
    16:07:09 🔄 '2106c935-…' 'A Dance with Dragons (2011)' ABS leads at 44.0972% (normalized: 77666.2s, source=abs)
    16:07:09 ✅ Storyteller API: 19120e99... -> 43.7% (TS: 1790194029919)

Guard 3 (`_peer_position_is_own_writeback`, #416) matched Storyteller's new
reading against BookBridge's last write by VALUE ONLY: |0.43665 - 0.43210| =
0.00455 <= SYNC_DELTA_BETWEEN_CLIENTS_PERCENT (default 1% -> 0.01), so it could
not distinguish "our write, re-read" from "the user moved less than 1% of the
book". The client poller's self-write check (CLIENT_POLLER_SELF_WRITE_ECHO_PERCENT)
has the identical blind spot.

Fix: Storyteller stores the `timestamp` a writer supplies verbatim and returns
it unchanged until a DIFFERENT write is accepted (verified against Storyteller
source v2.14.21 and v3.0.0-beta.38). BookBridge now records that value as an
opaque *marker* alongside every write; a later read is judged our own echo only
when the service still holds that exact marker, regardless of how close the
percentage happens to land. Clients that report no marker keep the pre-#447
value-only match unchanged (see write_tracker.marker_echo_verdict).
"""

import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.base_sync_test import BaseSyncCycleTestCase
from src.api.storyteller_api import StorytellerAPIClient
from src.services import write_tracker
import src.services.client_poller as client_poller_module
from src.services.client_poller import ClientPoller
from src.sync_clients.sync_client_interface import (
    LocatorResult,
    ServiceState,
    SyncResult,
    UpdateProgressRequest,
)
from src.sync_clients.storyteller_sync_client import StorytellerSyncClient
from src.sync_manager import SyncManager

# The reporter's own numbers, verbatim.
ABS_ID = "storyteller-marker-447"
TITLE = "A Dance with Dragons"
WRITTEN_PCT = 0.4366507206007024
WRITTEN_MARKER = 1790193848517
STORYTELLER_NOW_PCT = 0.4321
STORYTELLER_NOW_MARKER = 1790193990000
ABS_LEAD_PCT = 0.440972  # 44.0972%, the reporter's stationary ABS position


def _state(current: dict, previous_pct: float = 0.0, delta: float = 0.0) -> ServiceState:
    return ServiceState(
        current=current,
        previous_pct=previous_pct,
        delta=delta,
        threshold=0.01,
        is_configured=True,
        display=("X", "{prev:.2%}->{curr:.2%}"),
        value_formatter=lambda v: f"{v:.4%}",
    )


def _book447():
    return SimpleNamespace(duration=10000, transcript_file=None, sync_mode="audiobook")


def _clear_guard_env():
    for key in (
        "SYNC_FRESHNESS_GUARDS",
        "SYNC_ROLLBACK_VETO_SECONDS",
        "SYNC_PERIOD_MINS",
        "SYNC_DELTA_BETWEEN_CLIENTS_PERCENT",
    ):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# (a) / (b) -- integration: the real SyncManager.sync_cycle()
# ---------------------------------------------------------------------------


class _StorytellerMarkerSyncTestCase(BaseSyncCycleTestCase):
    """Shared scaffolding for the real-sync_cycle regression tests (a)/(b).

    Mirrors the reporter's prior cycle: BookBridge wrote WRITTEN_PCT to every
    follower and recorded a write-tracker entry for each (only Storyteller's
    carries a marker -- KoSync has none). This cycle, Storyteller reports
    STORYTELLER_NOW_PCT with a `position_ts` the test controls.
    """

    STORYTELLER_UUID = "st-uuid-447"

    def setUp(self):
        super().setUp()
        _clear_guard_env()
        self._saved_writes = dict(write_tracker._recent_writes)
        write_tracker._recent_writes.clear()
        self.addCleanup(self._restore)

    def _restore(self):
        write_tracker._recent_writes.clear()
        write_tracker._recent_writes.update(self._saved_writes)
        _clear_guard_env()

    def get_test_mapping(self):
        return {
            'abs_id': ABS_ID,
            'abs_title': TITLE,
            'kosync_doc_id': 'kosync-doc-447',
            'ebook_filename': 'test-book.epub',
            'storyteller_uuid': self.STORYTELLER_UUID,
            'transcript_file': str(Path(self.temp_dir) / 'test_transcript.json'),
            'status': 'active',
            'duration': 10000.0,
        }

    def get_test_state_data(self):
        # Storyteller's previous State is WRITTEN_PCT -- exactly what the prior
        # cycle wrote to it and (per sync_cycle's follower-state persistence)
        # what got saved back as its own reading.
        return {
            # 'ts' must be set: ABSSyncClient.get_service_state() computes its
            # delta against prev_state.timestamp, and a missing previous
            # timestamp crashes that fetch (caught upstream, but it silently
            # drops ABS out of the cycle instead of giving it a zero delta).
            'abs': {'pct': ABS_LEAD_PCT, 'ts': ABS_LEAD_PCT * 10000.0, 'last_updated': 1234567890},
            'storyteller': {'pct': WRITTEN_PCT, 'last_updated': 1234567890},
            'kosync': {'pct': 0.30, 'last_updated': 1234567890},
        }

    def get_expected_leader(self):
        return "Storyteller"

    def get_expected_final_percentage(self):
        return STORYTELLER_NOW_PCT

    def get_progress_mock_returns(self):
        return {
            'abs_progress': {'currentTime': ABS_LEAD_PCT * 10000.0, 'duration': 10000.0},
            'abs_in_progress': [{'id': ABS_ID, 'progress': ABS_LEAD_PCT, 'duration': 10000.0}],
            'kosync_progress': (0.30, "/body/DocFragment[1]/body/p[5]/text().0"),
            'storyteller_progress': (STORYTELLER_NOW_PCT, STORYTELLER_NOW_MARKER, "OEBPS/Text/chapter-23.xhtml", "frag-forward"),
            'booklore_progress': (0.0, None),
        }

    def _build_manager(self, storyteller_position_ts):
        mocks = self.setup_common_mocks()

        # A bare Mock's get_position_details_payload would return a Mock (not a
        # dict), which routes get_service_state through the legacy tuple
        # fallback path that carries no position_ts -- give it a real dict.
        mocks['storyteller_client'].get_position_details_payload.return_value = {
            "pct": STORYTELLER_NOW_PCT,
            "ts": STORYTELLER_NOW_MARKER,
            "position_ts": storyteller_position_ts,
            "href": "OEBPS/Text/chapter-23.xhtml",
            "frag": "frag-forward",
            "fragment": "frag-forward",
            "fragments": ["frag-forward"],
            "chapter_progress": 0.62,
            "css_selector": None,
            "position": None,
            "match_index": None,
            "cfi": None,
        }
        mocks['storyteller_client'].update_position.return_value = True

        # The leader's text gets relocated into each follower's own EPUB via
        # ebook_parser.find_text_location(epub, txt, hint_percentage=...); an
        # unconfigured Mock return value is not a real LocatorResult and
        # crashes the collapse-guard check regardless of which client leads.
        def _find_text_location(epub_file_name, txt, hint_percentage=None, **kwargs):
            pct = hint_percentage if hint_percentage is not None else 0.0
            return LocatorResult(percentage=pct, href="OEBPS/Text/chapter-01.xhtml", match_index=1)

        mocks['ebook_parser'].find_text_location.side_effect = _find_text_location

        transcriber = Mock()
        transcriber.get_text_at_time.return_value = "text"
        # The reader's Storyteller text resolves two minutes of audio past where
        # ABS stopped: they read on from the audiobook position.
        transcriber.find_time_for_text.return_value = ABS_LEAD_PCT * 10000.0 + 120.0

        from src.sync_clients.abs_sync_client import ABSSyncClient
        from src.sync_clients.kosync_sync_client import KoSyncSyncClient
        from src.sync_clients.abs_ebook_sync_client import ABSEbookSyncClient

        abs_sync_client = ABSSyncClient(mocks['abs_client'], transcriber, mocks['ebook_parser'])
        kosync_sync_client = KoSyncSyncClient(mocks['kosync_client'], mocks['ebook_parser'])
        abs_ebook_sync_client = ABSEbookSyncClient(mocks['abs_client'], mocks['ebook_parser'])
        storyteller_sync_client = StorytellerSyncClient(mocks['storyteller_client'], mocks['ebook_parser'])

        manager = SyncManager(
            abs_client=mocks['abs_client'],
            transcriber=transcriber,
            ebook_parser=mocks['ebook_parser'],
            database_service=mocks['database_service'],
            sync_clients={
                "ABS": abs_sync_client,
                "ABS eBook": abs_ebook_sync_client,
                "KoSync": kosync_sync_client,
                "Storyteller": storyteller_sync_client,
            },
            epub_cache_dir=Path(self.temp_dir) / 'epub_cache',
            data_dir=Path(self.temp_dir),
            books_dir=Path(self.temp_dir) / 'books',
        )
        manager.sync_clients['ABS']._update_abs_progress_with_offset = Mock(
            return_value=({"success": True}, ABS_LEAD_PCT * 10000.0)
        )
        manager._automatch_hardcover = Mock()
        manager._sync_to_hardcover = Mock()
        return manager, mocks

    @staticmethod
    def _seed_prior_cycle_writes():
        """Mirror the reporter's prior cycle: BookBridge wrote to every follower.
        Only Storyteller supplies a marker -- KoSync has none (#447 is additive)."""
        write_tracker.record_write("Storyteller", ABS_ID, WRITTEN_PCT, marker=WRITTEN_MARKER)
        write_tracker.record_write("KoSync", ABS_ID, 0.30)
        write_tracker.record_write("ABS eBook", ABS_ID, 0.0)


class TestStorytellerMarkerMismatchAllowsRealMovement(_StorytellerMarkerSyncTestCase):
    """(a) -- the marker changed: Storyteller is NOT our echo. MUST fail on the
    pre-fix code, which classifies this as an echo purely on value proximity."""

    def test_marker_mismatch_lets_storyteller_lead(self):
        self._seed_prior_cycle_writes()
        manager, mocks = self._build_manager(storyteller_position_ts=STORYTELLER_NOW_MARKER)

        with self.assertLogs("src.sync_manager", level="INFO") as captured:
            manager.sync_cycle()
        logs = "\n".join(captured.output)

        self.assertNotIn(
            "Ignoring 'Storyteller' delta (43.21%): it holds BookBridge's own write-back, "
            "not user movement",
            logs,
        )
        self.assertIn("Storyteller leads at", logs)
        mocks['storyteller_client'].update_position.assert_not_called()
        # The reader's Storyteller position is what gets propagated to ABS.
        self.assertTrue(manager.sync_clients['ABS']._update_abs_progress_with_offset.called)


class TestStorytellerMarkerMatchKeepsEchoBehavior(_StorytellerMarkerSyncTestCase):
    """(b) -- counterpart: the marker is UNCHANGED, so #416's echo behaviour
    still applies exactly as before."""

    def test_marker_match_keeps_echo_behavior(self):
        self._seed_prior_cycle_writes()
        manager, mocks = self._build_manager(storyteller_position_ts=WRITTEN_MARKER)

        with self.assertLogs("src.sync_manager", level="INFO") as captured:
            manager.sync_cycle()
        logs = "\n".join(captured.output)

        self.assertIn(
            "Ignoring 'Storyteller' delta (43.21%): it holds BookBridge's own write-back, "
            "not user movement",
            logs,
        )
        # Echo wins: ABS (the only remaining genuine candidate) leads and
        # Storyteller gets rewritten back, exactly as #416 already covers.
        mocks['storyteller_client'].update_position.assert_called_once()
        # That rewrite's timestamp must survive both recorders (the client's own
        # record_write and sync_manager's _record_bridge_write, which runs last)
        # as the marker the next cycle compares against.
        posted_timestamp = mocks['storyteller_client'].update_position.call_args.kwargs["timestamp"]
        recent = write_tracker.get_recent_write("Storyteller", ABS_ID, suppression_window=600)
        self.assertEqual(recent["marker"], posted_timestamp)


# ---------------------------------------------------------------------------
# (c) -- _determine_leader level (Guard 3 marker precedence)
# ---------------------------------------------------------------------------


def _leader_manager(delta_clients, client_names=("Storyteller", "ABS")):
    manager = SyncManager.__new__(SyncManager)

    class _Client:
        def can_be_leader(self):
            return True

    manager.sync_clients = {name: _Client() for name in client_names}
    manager._has_significant_delta = MagicMock(side_effect=lambda name, cfg, book: name in delta_clients)
    manager._normalize_for_cross_format_comparison = MagicMock(return_value=None)
    manager._get_primary_audio_client_name = MagicMock(return_value="ABS")
    manager.sync_delta_between_clients = 0.01
    manager.cross_format_deadband_seconds = 2.0
    return manager


class _MarkerLeaderBase(unittest.TestCase):
    """write_tracker holds module globals; save and restore them so this file
    cannot poison a suite that must pass in any order."""

    def setUp(self):
        _clear_guard_env()
        self._saved_writes = dict(write_tracker._recent_writes)
        write_tracker._recent_writes.clear()
        self.addCleanup(self._restore)

    def _restore(self):
        write_tracker._recent_writes.clear()
        write_tracker._recent_writes.update(self._saved_writes)
        _clear_guard_env()

    def _lead(self, manager, config):
        with self.assertLogs("src.sync_manager", level="INFO") as captured:
            leader, leader_pct = manager._determine_leader(config, _book447(), ABS_ID, TITLE)
        return leader, leader_pct, "\n".join(captured.output)


class TestGuard3MarkerPrecedence(_MarkerLeaderBase):
    def test_marker_changed_storyteller_not_excluded_and_leads(self):
        write_tracker.record_write("Storyteller", ABS_ID, WRITTEN_PCT, marker=WRITTEN_MARKER)
        config = {
            "Storyteller": _state(
                {"pct": STORYTELLER_NOW_PCT, "_position_marker": STORYTELLER_NOW_MARKER},
                previous_pct=WRITTEN_PCT,
            ),
            "ABS": _state({"pct": ABS_LEAD_PCT}, previous_pct=ABS_LEAD_PCT),
        }
        manager = _leader_manager(delta_clients=("Storyteller",))

        leader, leader_pct, logs = self._lead(manager, config)

        self.assertEqual(leader, "Storyteller")
        self.assertAlmostEqual(leader_pct, STORYTELLER_NOW_PCT)
        self.assertNotIn("Ignoring 'Storyteller' delta", logs)

    def test_marker_equal_excludes_storyteller(self):
        write_tracker.record_write("Storyteller", ABS_ID, WRITTEN_PCT, marker=WRITTEN_MARKER)
        config = {
            "Storyteller": _state(
                {"pct": STORYTELLER_NOW_PCT, "_position_marker": WRITTEN_MARKER},
                previous_pct=WRITTEN_PCT,
            ),
            "ABS": _state({"pct": ABS_LEAD_PCT}, previous_pct=ABS_LEAD_PCT),
        }
        manager = _leader_manager(delta_clients=("Storyteller",))

        leader, leader_pct, logs = self._lead(manager, config)

        self.assertIn(
            "Ignoring 'Storyteller' delta (43.21%): it holds BookBridge's own write-back, "
            "not user movement",
            logs,
        )
        self.assertNotEqual(leader, "Storyteller")

    def test_no_observed_marker_falls_back_to_value_match(self):
        """Storyteller reports no marker at all (a legacy/degraded read) -- Guard
        3 falls back to the pre-#447 value match exactly as it worked before."""
        write_tracker.record_write("Storyteller", ABS_ID, WRITTEN_PCT, marker=WRITTEN_MARKER)
        config = {
            "Storyteller": _state({"pct": STORYTELLER_NOW_PCT}, previous_pct=WRITTEN_PCT),
            "ABS": _state({"pct": ABS_LEAD_PCT}, previous_pct=ABS_LEAD_PCT),
        }
        manager = _leader_manager(delta_clients=("Storyteller",))

        leader, leader_pct, logs = self._lead(manager, config)

        self.assertIn(
            "Ignoring 'Storyteller' delta (43.21%): it holds BookBridge's own write-back, "
            "not user movement",
            logs,
        )
        self.assertNotEqual(leader, "Storyteller")


# ---------------------------------------------------------------------------
# (d) -- write_tracker unit tests
# ---------------------------------------------------------------------------


class TestWriteTrackerMarker(unittest.TestCase):
    def setUp(self):
        self._saved_writes = dict(write_tracker._recent_writes)
        write_tracker._recent_writes.clear()

    def tearDown(self):
        write_tracker._recent_writes.clear()
        write_tracker._recent_writes.update(self._saved_writes)

    def test_marker_round_trips(self):
        write_tracker.record_write("Storyteller", "book-1", pct=STORYTELLER_NOW_PCT, marker=STORYTELLER_NOW_MARKER)
        meta = write_tracker.get_recent_write("Storyteller", "book-1")
        self.assertEqual(meta["marker"], STORYTELLER_NOW_MARKER)

    def test_marker_defaults_to_none(self):
        write_tracker.record_write("KoSync", "book-2", pct=0.5)
        meta = write_tracker.get_recent_write("KoSync", "book-2")
        self.assertIsNone(meta["marker"])

    def test_legacy_two_tuple_entry_reads_marker_none(self):
        """An entry written before #447 (a bare (ts, pct) 2-tuple) must still
        read back cleanly with marker=None, not raise."""
        write_tracker.record_write("KoSync", "book-3", pct=0.5)
        key = write_tracker._key("KoSync", "book-3", None)
        write_tracker._recent_writes[key] = write_tracker._recent_writes[key][:2]
        meta = write_tracker.get_recent_write("KoSync", "book-3")
        self.assertIsNotNone(meta)
        self.assertIsNone(meta["marker"])


class TestMarkerEchoVerdict(unittest.TestCase):
    def setUp(self):
        self._saved_writes = dict(write_tracker._recent_writes)
        write_tracker._recent_writes.clear()

    def tearDown(self):
        write_tracker._recent_writes.clear()
        write_tracker._recent_writes.update(self._saved_writes)

    def test_no_recent_write_is_undecidable(self):
        self.assertIsNone(write_tracker.marker_echo_verdict(None, WRITTEN_MARKER))

    def test_recent_with_no_marker_is_undecidable(self):
        recent = {"ts": 0.0, "age": 0.0, "pct": 0.5, "marker": None}
        self.assertIsNone(write_tracker.marker_echo_verdict(recent, WRITTEN_MARKER))

    def test_observed_with_no_marker_is_undecidable(self):
        recent = {"ts": 0.0, "age": 0.0, "pct": 0.5, "marker": WRITTEN_MARKER}
        self.assertIsNone(write_tracker.marker_echo_verdict(recent, None))

    def test_equal_markers_are_echo(self):
        recent = {"ts": 0.0, "age": 0.0, "pct": 0.5, "marker": WRITTEN_MARKER}
        self.assertTrue(write_tracker.marker_echo_verdict(recent, WRITTEN_MARKER))

    def test_unequal_markers_are_not_echo(self):
        recent = {"ts": 0.0, "age": 0.0, "pct": 0.5, "marker": WRITTEN_MARKER}
        self.assertFalse(write_tracker.marker_echo_verdict(recent, STORYTELLER_NOW_MARKER))

    def test_user_scoping_is_the_callers_responsibility(self):
        """marker_echo_verdict only compares the two markers it is handed --
        get_recent_write has already resolved the user-scoped `recent` dict (or
        None) before this is ever called."""
        write_tracker.record_write("Storyteller", "book-4", pct=WRITTEN_PCT, user_id=1, marker=WRITTEN_MARKER)
        recent_for_user_1 = write_tracker.get_recent_write("Storyteller", "book-4", user_id=1)
        recent_for_user_2 = write_tracker.get_recent_write("Storyteller", "book-4", user_id=2)

        self.assertTrue(write_tracker.marker_echo_verdict(recent_for_user_1, WRITTEN_MARKER))
        self.assertIsNone(recent_for_user_2)
        self.assertIsNone(write_tracker.marker_echo_verdict(recent_for_user_2, WRITTEN_MARKER))


# ---------------------------------------------------------------------------
# (e) -- _peer_position_is_own_writeback + rollback veto marker precedence
# ---------------------------------------------------------------------------


class TestPeerPositionIsOwnWriteback(unittest.TestCase):
    def setUp(self):
        _clear_guard_env()
        self._saved_writes = dict(write_tracker._recent_writes)
        write_tracker._recent_writes.clear()
        self.manager = SyncManager.__new__(SyncManager)
        self.addCleanup(self._restore)

    def _restore(self):
        write_tracker._recent_writes.clear()
        write_tracker._recent_writes.update(self._saved_writes)
        _clear_guard_env()

    def test_marker_mismatch_inside_margin_is_not_echo_and_logs(self):
        write_tracker.record_write("Storyteller", "book-e1", WRITTEN_PCT, marker=WRITTEN_MARKER)

        with self.assertLogs("src.sync_manager", level="INFO") as captured:
            result = self.manager._peer_position_is_own_writeback(
                "book-e1", "Storyteller", STORYTELLER_NOW_PCT, 0.01,
                observed_marker=STORYTELLER_NOW_MARKER, title_snip="Book",
            )
        logs = "\n".join(captured.output)

        self.assertFalse(result)
        self.assertIn(
            "is within 1.00% of BookBridge's own write-back (43.67%), but the "
            "service's position marker changed since that write - treating it as "
            "user movement",
            logs,
        )

    def test_marker_match_outside_margin_is_still_echo(self):
        write_tracker.record_write("Storyteller", "book-e2", WRITTEN_PCT, marker=WRITTEN_MARKER)

        result = self.manager._peer_position_is_own_writeback(
            "book-e2", "Storyteller", 0.10, 0.01,
            observed_marker=WRITTEN_MARKER, title_snip="Book",
        )

        self.assertTrue(result)

    def test_marker_on_recorded_side_only_falls_back_to_value(self):
        write_tracker.record_write("Storyteller", "book-e3", WRITTEN_PCT, marker=WRITTEN_MARKER)

        result = self.manager._peer_position_is_own_writeback(
            "book-e3", "Storyteller", STORYTELLER_NOW_PCT, 0.01,
            observed_marker=None, title_snip="Book",
        )

        self.assertTrue(result)  # value match still holds (0.455% <= 1%)

    def test_marker_on_observed_side_only_falls_back_to_value(self):
        write_tracker.record_write("Storyteller", "book-e4", WRITTEN_PCT)  # no marker recorded

        result = self.manager._peer_position_is_own_writeback(
            "book-e4", "Storyteller", STORYTELLER_NOW_PCT, 0.01,
            observed_marker=STORYTELLER_NOW_MARKER, title_snip="Book",
        )

        self.assertTrue(result)  # value match still holds


class TestRollbackVetoMarkerPrecedence(_MarkerLeaderBase):
    """A peer whose marker changed is real evidence again; the rollback veto
    (#413) must not be fooled by a percentage that still happens to match our
    last write -- the veto against the behind client still fires."""

    CANDIDATE_TS = 1751400000.0
    PEER_SKEW_SECONDS = 93021.0
    PEER_PCT = 0.397525  # what we wrote to Storyteller, and what it still
                          # reports now, unchanged -- a pre-#447 value match
                          # would have called this our echo
    CANDIDATE_PCT = 0.227911  # ABS's genuine rewind, materially behind

    def test_marker_changed_peer_keeps_the_veto_despite_matching_value(self):
        write_tracker.record_write("Storyteller", ABS_ID, self.PEER_PCT, marker=WRITTEN_MARKER)

        manager = _leader_manager(delta_clients=("ABS",), client_names=("ABS", "Storyteller"))
        manager._get_primary_audio_client_name = MagicMock(return_value="ABS")

        config = {
            "ABS": _state({
                "pct": self.CANDIDATE_PCT,
                "service_updated_at": self.CANDIDATE_TS,
                "_service_prev_updated_at": self.CANDIDATE_TS - 3600.0,
            }, previous_pct=self.PEER_PCT),
            "Storyteller": _state({
                "pct": self.PEER_PCT,
                # The marker moved on since our last write -- a genuine new
                # read, even though the raw percentage is unchanged.
                "_position_marker": STORYTELLER_NOW_MARKER,
                "service_updated_at": self.CANDIDATE_TS + self.PEER_SKEW_SECONDS,
                "_service_prev_updated_at": self.CANDIDATE_TS,
            }, previous_pct=self.PEER_PCT),
        }

        leader, leader_pct, logs = self._lead(manager, config)

        self.assertEqual(leader, "Storyteller")
        self.assertEqual(leader_pct, self.PEER_PCT)
        self.assertIn("Rollback veto: 'ABS'", logs)
        self.assertNotIn("Rollback veto skipped", logs)


# ---------------------------------------------------------------------------
# (f) -- _record_bridge_write marker forwarding
# ---------------------------------------------------------------------------


class TestRecordBridgeWriteMarkerForwarding(unittest.TestCase):
    def setUp(self):
        self._saved_writes = dict(write_tracker._recent_writes)
        write_tracker._recent_writes.clear()
        self.manager = SyncManager.__new__(SyncManager)

    def tearDown(self):
        write_tracker._recent_writes.clear()
        write_tracker._recent_writes.update(self._saved_writes)

    @staticmethod
    def _marker(client_name, abs_id):
        recent = write_tracker.get_recent_write(client_name, abs_id, suppression_window=600)
        return recent["marker"] if recent else None

    def test_forwards_the_position_marker(self):
        self.manager._record_bridge_write(
            "Storyteller", "abs-447",
            SyncResult(STORYTELLER_NOW_PCT, True, {
                "pct": STORYTELLER_NOW_PCT, "_position_marker": STORYTELLER_NOW_MARKER,
            }),
        )
        self.assertEqual(self._marker("Storyteller", "abs-447"), STORYTELLER_NOW_MARKER)

    def test_absent_marker_records_none(self):
        self.manager._record_bridge_write(
            "KoSync", "abs-447", SyncResult(0.5, True, {"pct": 0.5}),
        )
        self.assertIsNone(self._marker("KoSync", "abs-447"))

    def test_failed_write_records_nothing(self):
        self.manager._record_bridge_write(
            "Storyteller", "abs-447",
            SyncResult(STORYTELLER_NOW_PCT, False, {
                "pct": STORYTELLER_NOW_PCT, "_position_marker": STORYTELLER_NOW_MARKER,
            }),
        )
        recent = write_tracker.get_recent_write("Storyteller", "abs-447", suppression_window=600)
        self.assertIsNone(recent)


# ---------------------------------------------------------------------------
# (g) -- storyteller_api: position_ts derivation + update_position timestamp
# ---------------------------------------------------------------------------


class TestStorytellerPositionMarkerPayload(unittest.TestCase):
    def setUp(self):
        self.client = StorytellerAPIClient(credentials={
            "STORYTELLER_API_URL": "http://storyteller:8001",
            "STORYTELLER_USER": "reader",
            "STORYTELLER_PASSWORD": "secret",
        })

    def test_position_ts_from_integer_timestamp(self):
        with patch.object(self.client, "_make_request") as mock_request:
            mock_request.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"timestamp": STORYTELLER_NOW_MARKER, "locator": {}}),
            )
            payload = self.client.get_position_details_payload("book-uuid")

        self.assertEqual(payload["position_ts"], STORYTELLER_NOW_MARKER)

    def test_position_ts_from_whole_float_json_timestamp(self):
        with patch.object(self.client, "_make_request") as mock_request:
            mock_request.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"timestamp": float(STORYTELLER_NOW_MARKER), "locator": {}}),
            )
            payload = self.client.get_position_details_payload("book-uuid")

        self.assertEqual(payload["position_ts"], STORYTELLER_NOW_MARKER)
        self.assertIsInstance(payload["position_ts"], int)

    def test_position_ts_none_when_only_updated_at_present(self):
        """updatedAt is server-generated and never equals a timestamp we sent --
        it must never seed the marker."""
        with patch.object(self.client, "_make_request") as mock_request:
            mock_request.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"updatedAt": "2026-07-06T20:00:00Z", "locator": {}}),
            )
            payload = self.client.get_position_details_payload("book-uuid")

        self.assertIsNone(payload["position_ts"])
        self.assertGreater(payload["ts"], 0)  # "ts" still falls back to updatedAt

    def test_position_ts_none_when_timestamp_missing(self):
        with patch.object(self.client, "_make_request") as mock_request:
            mock_request.return_value = Mock(status_code=200, json=Mock(return_value={"locator": {}}))
            payload = self.client.get_position_details_payload("book-uuid")

        self.assertIsNone(payload["position_ts"])

    def test_update_position_posts_the_given_timestamp(self):
        locator = LocatorResult(percentage=0.64, href="text/chapter02.xhtml")
        with patch.object(self.client, "_make_request") as mock_request:
            mock_request.return_value = Mock(status_code=204)
            ok = self.client.update_position("book-uuid", 0.64, locator, timestamp=STORYTELLER_NOW_MARKER)

        self.assertTrue(ok)
        _, _, payload = mock_request.call_args.args
        self.assertEqual(payload["timestamp"], STORYTELLER_NOW_MARKER)

    def test_update_position_without_timestamp_still_posts_a_current_epoch_ms_int(self):
        locator = LocatorResult(percentage=0.64, href="text/chapter02.xhtml")
        with patch.object(self.client, "_make_request") as mock_request:
            mock_request.return_value = Mock(status_code=204)
            before = int(time.time() * 1000)
            ok = self.client.update_position("book-uuid", 0.64, locator)
            after = int(time.time() * 1000)

        self.assertTrue(ok)
        _, _, payload = mock_request.call_args.args
        self.assertIsInstance(payload["timestamp"], int)
        self.assertTrue(before <= payload["timestamp"] <= after)


# ---------------------------------------------------------------------------
# (h) -- storyteller sync client marker plumbing
# ---------------------------------------------------------------------------


class TestStorytellerSyncClientPositionMarker(unittest.TestCase):
    def setUp(self):
        self._saved_writes = dict(write_tracker._recent_writes)
        write_tracker._recent_writes.clear()

    def tearDown(self):
        write_tracker._recent_writes.clear()
        write_tracker._recent_writes.update(self._saved_writes)

    def test_get_service_state_sets_position_marker_from_payload(self):
        storyteller_api = MagicMock()
        storyteller_api.is_configured.return_value = True
        storyteller_api.get_position_details_payload.return_value = {
            "pct": STORYTELLER_NOW_PCT,
            "ts": STORYTELLER_NOW_MARKER,
            "position_ts": STORYTELLER_NOW_MARKER,
            "href": "chapter.xhtml",
        }
        client = StorytellerSyncClient(storyteller_api, MagicMock())
        book = SimpleNamespace(storyteller_uuid="uuid-h1")

        state = client.get_service_state(book, None)

        self.assertEqual(state.current["_position_marker"], STORYTELLER_NOW_MARKER)

    def test_get_service_state_omits_marker_when_position_ts_absent(self):
        storyteller_api = MagicMock()
        storyteller_api.is_configured.return_value = True
        storyteller_api.get_position_details_payload.return_value = {
            "pct": STORYTELLER_NOW_PCT,
            "ts": STORYTELLER_NOW_MARKER,
            "href": "chapter.xhtml",
        }
        client = StorytellerSyncClient(storyteller_api, MagicMock())
        book = SimpleNamespace(storyteller_uuid="uuid-h2")

        state = client.get_service_state(book, None)

        self.assertNotIn("_position_marker", state.current)

    def test_get_service_state_tuple_fallback_carries_no_marker(self):
        storyteller_api = MagicMock()
        storyteller_api.is_configured.return_value = True
        storyteller_api.get_position_details_payload.return_value = None
        storyteller_api.get_position_details_rich.return_value = (
            STORYTELLER_NOW_PCT, STORYTELLER_NOW_MARKER, "chapter.xhtml", "frag-1", 0.5,
        )
        client = StorytellerSyncClient(storyteller_api, MagicMock())
        book = SimpleNamespace(storyteller_uuid="uuid-h3")

        state = client.get_service_state(book, None)

        self.assertNotIn("_position_marker", state.current)

    def test_update_progress_uses_one_timestamp_for_post_marker_and_state(self):
        storyteller_api = MagicMock()
        storyteller_api.update_position.return_value = True
        client = StorytellerSyncClient(storyteller_api, MagicMock())

        book = SimpleNamespace(
            abs_id="abs-h4", abs_title="Test", ebook_filename="b.epub", storyteller_uuid="uuid-h4",
        )
        locator = LocatorResult(percentage=0.55, href="chapter.xhtml", fragment="frag-1")
        request = UpdateProgressRequest(locator_result=locator, txt=None)

        result = client.update_progress(book, request)

        self.assertTrue(result.success)
        posted_timestamp = storyteller_api.update_position.call_args.kwargs["timestamp"]
        self.assertIsInstance(posted_timestamp, int)
        recorded = write_tracker.get_recent_write("Storyteller", "abs-h4", suppression_window=600)
        self.assertEqual(recorded["marker"], posted_timestamp)
        self.assertEqual(result.updated_state["_position_marker"], posted_timestamp)

    def test_failed_post_records_nothing_and_no_marker_in_updated_state(self):
        storyteller_api = MagicMock()
        storyteller_api.update_position.return_value = False
        client = StorytellerSyncClient(storyteller_api, MagicMock())

        book = SimpleNamespace(
            abs_id="abs-h5", abs_title="Test", ebook_filename="b.epub", storyteller_uuid="uuid-h5",
        )
        locator = LocatorResult(percentage=0.55, href="chapter.xhtml", fragment="frag-1")
        request = UpdateProgressRequest(locator_result=locator, txt=None)

        result = client.update_progress(book, request)

        self.assertFalse(result.success)
        self.assertNotIn("_position_marker", result.updated_state)
        self.assertIsNone(write_tracker.get_recent_write("Storyteller", "abs-h5", suppression_window=600))


# ---------------------------------------------------------------------------
# (i) -- client_poller marker precedence
# ---------------------------------------------------------------------------


class _ImmediateThread447:
    def __init__(self, target=None, kwargs=None, daemon=None):
        self._target = target
        self._kwargs = kwargs or {}

    def start(self):
        if self._target:
            self._target(**self._kwargs)


class TestClientPollerMarkerPrecedence(unittest.TestCase):
    def setUp(self):
        with write_tracker._writes_lock:
            self._saved_writes = dict(write_tracker._recent_writes)
            write_tracker._recent_writes.clear()
        self._thread_patch = patch.object(client_poller_module.threading, "Thread", _ImmediateThread447)
        self._thread_patch.start()
        self.addCleanup(self._thread_patch.stop)
        self.addCleanup(self._restore_write_tracker)

    def _restore_write_tracker(self):
        with write_tracker._writes_lock:
            write_tracker._recent_writes.clear()
            write_tracker._recent_writes.update(self._saved_writes)

    @staticmethod
    def _poller(current_state_extra):
        book = SimpleNamespace(abs_id="abs-i1", abs_title="Book I")
        db = MagicMock()
        db.get_books_by_status.return_value = [book]
        sync_manager = MagicMock()
        sync_client = MagicMock()
        sync_client.is_configured.return_value = True
        current = {"pct": STORYTELLER_NOW_PCT}
        current.update(current_state_extra)
        sync_client.get_service_state.return_value = SimpleNamespace(current=current)
        poller = ClientPoller(db, sync_manager, {"Storyteller": sync_client})
        return poller, sync_manager, sync_client

    def test_marker_changed_triggers_sync_even_within_echo_value_window(self):
        write_tracker.record_write("Storyteller", "abs-i1", WRITTEN_PCT, marker=WRITTEN_MARKER)
        poller, sync_manager, sync_client = self._poller({"_position_marker": STORYTELLER_NOW_MARKER})
        # Forces marker_changed=True regardless of fingerprint details, matching
        # the existing self-write tests' convention (test_client_poller_self_write.py).
        poller._last_known[(None, "Storyteller", "abs-i1")] = WRITTEN_PCT

        poller._poll_client("Storyteller")

        sync_manager.sync_cycle.assert_called_once_with(target_abs_id="abs-i1", user_id=None)

    def test_marker_equal_is_ignored_in_marker_changed_branch(self):
        write_tracker.record_write("Storyteller", "abs-i1", WRITTEN_PCT, marker=WRITTEN_MARKER)
        poller, sync_manager, sync_client = self._poller({"_position_marker": WRITTEN_MARKER})
        poller._last_known[(None, "Storyteller", "abs-i1")] = WRITTEN_PCT

        poller._poll_client("Storyteller")

        sync_manager.sync_cycle.assert_not_called()

    def test_settled_branch_marker_changed_triggers_sync(self):
        write_tracker.record_write("Storyteller", "abs-i1", WRITTEN_PCT, marker=WRITTEN_MARKER)
        poller, sync_manager, sync_client = self._poller({"_position_marker": STORYTELLER_NOW_MARKER})
        current = sync_client.get_service_state.return_value.current
        poller._last_known[(None, "Storyteller", "abs-i1")] = poller._state_fingerprint(current)
        poller._pending_sync[(None, "Storyteller", "abs-i1")] = STORYTELLER_NOW_PCT

        poller._poll_client("Storyteller")

        sync_manager.sync_cycle.assert_called_once_with(target_abs_id="abs-i1", user_id=None)
        self.assertNotIn((None, "Storyteller", "abs-i1"), poller._pending_sync)

    def test_settled_branch_marker_equal_stays_suppressed(self):
        write_tracker.record_write("Storyteller", "abs-i1", WRITTEN_PCT, marker=WRITTEN_MARKER)
        poller, sync_manager, sync_client = self._poller({"_position_marker": WRITTEN_MARKER})
        current = sync_client.get_service_state.return_value.current
        poller._last_known[(None, "Storyteller", "abs-i1")] = poller._state_fingerprint(current)
        poller._pending_sync[(None, "Storyteller", "abs-i1")] = STORYTELLER_NOW_PCT

        poller._poll_client("Storyteller")

        sync_manager.sync_cycle.assert_not_called()
        self.assertNotIn((None, "Storyteller", "abs-i1"), poller._pending_sync)


if __name__ == "__main__":
    unittest.main()
