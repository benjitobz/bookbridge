"""Tests for the KoSync device arbiter (issue #215, KOSYNC_ACTIVE_DEVICE_WINS).

The rule under test: when one book is linked through several reader files, a device
that PROVES it is being read — same device, more than one report, advancing — beats a
furthest-read row nobody has touched. Without proof, furthest-wins is unchanged, so a
stale second reader still cannot drag progress backwards.

Shadow mode is the default. `choose_device_row` reports the same verdict in shadow and
on; only the GET handler decides whether to act on it, which is what lets the decision
be judged against real traffic before it changes what a reader receives.
"""

import os
import unittest
from datetime import timedelta
from types import SimpleNamespace

from src.services import kosync_device_arbiter, observation_trail
from src.utils.config_loader import ALL_SETTINGS, DEFAULT_CONFIG
from src.utils.progress_metadata import parse_service_timestamp
from src.utils.time_utils import utcnow

ABS_ID = "abs-arbiter-1"
USER_ID = 1


def row(device, pct, minutes_ago, *, device_id=None, doc_hash=None):
    return SimpleNamespace(
        document_hash=doc_hash or f"hash-{device}",
        percentage=pct,
        progress=f"/body/{device}-{int(pct * 100)}.0",
        device=device,
        device_id=device_id or f"id-{device}",
        timestamp=utcnow() - timedelta(minutes=minutes_ago),
        user_id=USER_ID,
    )


def choose(rows, **kwargs):
    return kosync_device_arbiter.choose_device_row(
        rows, abs_id=ABS_ID, user_id=USER_ID,
        parse_timestamp=parse_service_timestamp, **kwargs,
    )


class TestDeviceArbiter(unittest.TestCase):
    def setUp(self):
        observation_trail.clear()
        self._saved = os.environ.get("KOSYNC_ACTIVE_DEVICE_WINS")
        os.environ["KOSYNC_ACTIVE_DEVICE_WINS"] = "on"

    def tearDown(self):
        observation_trail.clear()
        if self._saved is None:
            os.environ.pop("KOSYNC_ACTIVE_DEVICE_WINS", None)
        else:
            os.environ["KOSYNC_ACTIVE_DEVICE_WINS"] = self._saved

    def _reading(self, device, *pcts):
        for pct in pcts:
            observation_trail.record_observation(
                "KoSync", ABS_ID, pct, source="put", user_id=USER_ID, device=device,
            )

    def test_active_device_beats_an_older_furthest_row(self):
        """The case from the report: reading on the phone, dragged to the Kobo's 86%."""
        self._reading("phone", 0.40, 0.41, 0.42)
        choice = choose([row("kobo", 0.86, 60), row("phone", 0.42, 1)])
        self.assertTrue(choice.overrides_furthest)
        self.assertEqual(choice.active_device, "phone")
        self.assertEqual(choice.row.device, "phone")
        self.assertEqual(choice.furthest_row.device, "kobo")

    def test_without_proof_the_furthest_row_still_wins(self):
        """One reading is what a stale row looks like — it proves nothing."""
        self._reading("phone", 0.42)
        choice = choose([row("kobo", 0.86, 60), row("phone", 0.42, 1)])
        self.assertFalse(choice.overrides_furthest)
        self.assertEqual(choice.row.device, "kobo")
        self.assertIn("no device corroborated", choice.reason)

    def test_a_newer_furthest_row_is_not_the_stale_sibling_case(self):
        """If the furthest row is fresher than the active reader, leave it alone."""
        self._reading("phone", 0.40, 0.41, 0.42)
        choice = choose([row("kobo", 0.86, 1), row("phone", 0.42, 60)])
        self.assertFalse(choice.overrides_furthest)
        self.assertIn("not older", choice.reason)

    def test_two_active_devices_are_a_real_conflict_and_are_not_settled_here(self):
        self._reading("phone", 0.40, 0.41, 0.42)
        self._reading("kobo", 0.80, 0.85, 0.86)
        choice = choose([row("kobo", 0.86, 60), row("phone", 0.42, 1)])
        self.assertFalse(choice.overrides_furthest)
        self.assertIn("2 devices active", choice.reason)

    def test_one_device_is_never_arbitrated(self):
        self._reading("phone", 0.40, 0.41, 0.42)
        choice = choose([row("phone", 0.42, 1)])
        self.assertFalse(choice.overrides_furthest)
        self.assertIn("one device", choice.reason)

    def test_internal_sync_bot_rows_are_not_a_second_device(self):
        """Our own write-back wearing a device name must not create a disagreement."""
        self._reading("phone", 0.40, 0.41, 0.42)
        rows = [
            row("abs-sync-bot", 0.86, 60, device_id="abs-sync-bot"),
            row("phone", 0.42, 1),
        ]
        choice = choose(
            rows,
            is_internal_device=lambda d, did: "abs-sync-bot" in (str(d or ""), str(did or "")),
        )
        self.assertFalse(choice.overrides_furthest)
        self.assertIn("one device", choice.reason)

    def test_off_returns_furthest_without_consulting_the_trail(self):
        self._reading("phone", 0.40, 0.41, 0.42)
        os.environ["KOSYNC_ACTIVE_DEVICE_WINS"] = "off"
        choice = choose([row("kobo", 0.86, 60), row("phone", 0.42, 1)])
        self.assertFalse(choice.overrides_furthest)
        self.assertIn("arbiter off", choice.reason)

    def test_shadow_reports_the_same_verdict_as_on(self):
        """Shadow has to compute the real decision — that is the whole point of it."""
        self._reading("phone", 0.40, 0.41, 0.42)
        rows = [row("kobo", 0.86, 60), row("phone", 0.42, 1)]
        os.environ["KOSYNC_ACTIVE_DEVICE_WINS"] = "on"
        live = choose(rows)
        os.environ["KOSYNC_ACTIVE_DEVICE_WINS"] = "shadow"
        shadow = choose(rows)
        self.assertEqual(live.overrides_furthest, shadow.overrides_furthest)
        self.assertEqual(live.reason, shadow.reason)

    def test_an_unrecognized_mode_falls_back_to_shadow(self):
        """A typo must never silently change what a reader receives."""
        os.environ["KOSYNC_ACTIVE_DEVICE_WINS"] = "yes-please"
        self.assertEqual(kosync_device_arbiter.arbiter_mode(), kosync_device_arbiter.MODE_SHADOW)

    def test_no_rows_with_progress_yields_no_choice(self):
        self.assertIsNone(choose([]))

    def test_setting_is_registered_with_a_shadow_default(self):
        self.assertIn("KOSYNC_ACTIVE_DEVICE_WINS", ALL_SETTINGS)
        self.assertEqual(DEFAULT_CONFIG["KOSYNC_ACTIVE_DEVICE_WINS"], "shadow")


class FakeDatabase:
    def __init__(self, state, rows):
        self.state = state
        self.rows = rows

    def get_states_for_book(self, _abs_id):
        return [self.state] if self.state is not None else []

    def get_user_kosync_progress_for_book(self, _abs_id, user_id):
        return [r for r in self.rows if r.user_id == user_id]

    def get_kosync_documents_for_book(self, _abs_id):
        return self.rows


class TestArbiterWiredIntoTheGetPath(unittest.TestCase):
    """The GET handler must actually act on the verdict — and only when mode is on.

    The synced State deliberately sits at the stale row's position, which is what it
    holds in the real failure: the last cycle took 86% from the GET and wrote it back.
    That is why the proven reader can never be "ahead", and why the arbiter has to
    answer directly instead of feeding the furthest-wins gate.
    """

    def setUp(self):
        observation_trail.clear()
        self._saved = os.environ.get("KOSYNC_ACTIVE_DEVICE_WINS")

    def tearDown(self):
        observation_trail.clear()
        if self._saved is None:
            os.environ.pop("KOSYNC_ACTIVE_DEVICE_WINS", None)
        else:
            os.environ["KOSYNC_ACTIVE_DEVICE_WINS"] = self._saved

    def _respond(self, mode):
        from flask import Flask, g
        from src.api import kosync_server

        os.environ["KOSYNC_ACTIVE_DEVICE_WINS"] = mode
        for pct in (0.40, 0.41, 0.42):
            observation_trail.record_observation(
                "KoSync", ABS_ID, pct, source="put", user_id=USER_ID, device="phone",
            )
        kobo = row("kobo", 0.86, 60, doc_hash="k" * 32)
        phone = row("phone", 0.42, 1, doc_hash="p" * 32)
        state = SimpleNamespace(
            client_name="kosync", percentage=0.86, xpath="/body/kobo-86.0", cfi="",
            last_updated=1_700_000_000.0, locator_json=None, user_id=USER_ID,
        )
        book = SimpleNamespace(
            abs_id=ABS_ID, abs_title="Arbiter Book", kosync_doc_id="p" * 32,
            sync_mode="ebook", ebook_filename="b.epub", original_ebook_filename="b.epub",
        )
        old_db = kosync_server._database_service
        kosync_server._database_service = FakeDatabase(state, [kobo, phone])
        try:
            app = Flask(__name__)
            with app.test_request_context("/"):
                g.kosync_user_id = USER_ID
                response, status = kosync_server._respond_from_book_states("p" * 32, book)
                return response.get_json(), status
        finally:
            kosync_server._database_service = old_db

    def test_on_hands_back_the_device_being_read(self):
        data, status = self._respond("on")
        self.assertEqual(status, 200)
        self.assertAlmostEqual(data["percentage"], 0.42)
        self.assertEqual(data["progress"], "/body/phone-42.0")

    def test_shadow_changes_nothing_the_reader_receives(self):
        data, status = self._respond("shadow")
        self.assertEqual(status, 200)
        self.assertAlmostEqual(data["percentage"], 0.86)

    def test_off_changes_nothing_the_reader_receives(self):
        data, status = self._respond("off")
        self.assertEqual(status, 200)
        self.assertAlmostEqual(data["percentage"], 0.86)


if __name__ == "__main__":
    unittest.main()
