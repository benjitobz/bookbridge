"""A second KOReader device must be able to prove a deliberate rewind (issue #215).

Reproduces a live capture from 2026-09-18. Both devices held byte-identical copies of
the book — which is the NORMAL case when BridgeSync serves the file to both — so they
shared one document hash and one `kosync_user_progress` row. The Kobo had read to
47.3%; the Kindle was paged back and read on:

    KOSync: PUT progress request for doc 034ab404… (device: KindlePaperWhite5SE)
    KOSync: Ignored progress from 'KindlePaperWhite5SE' ... (user has higher: 0.47% vs new 0.27%)
    KOSync: Ignored progress from 'KindlePaperWhite5SE' ... (user has higher: 0.47% vs new 0.29%)
    KOSync: Ignored progress from 'KindlePaperWhite5SE' ... (user has higher: 0.47% vs new 0.30%)

Three reports, advancing, over 63 seconds — a textbook corroborated rewind, rejected
every time. The cause was ordering: furthest-wins returned before the observation was
recorded, so proving the rewind required observations, observations required accepted
PUTs, and the PUTs were rejected for being the rewind. Nothing the reader did could
ever escape it.

`same_device` is why one device never hit this: it skips the gate entirely.
"""

import os
import unittest

from src.api import kosync_server
from src.services import observation_trail

ABS_ID = "bookorbit:test-apex"
DOC_HASH = "034ab4042ae8e0385c225fe0c078aaa2"
KOBO = ("Kobo_monza", "5C299DF960C94850BE41")
KINDLE = ("KindlePaperWhite5SE", "DA1CE13860D0456384A5")


class _Doc:
    def __init__(self, linked_abs_id):
        self.linked_abs_id = linked_abs_id


class TestSecondDeviceRewindCorroboration(unittest.TestCase):
    def setUp(self):
        observation_trail.clear()
        self._saved = os.environ.get("SYNC_TRUST_CORROBORATED_REWIND")
        os.environ["SYNC_TRUST_CORROBORATED_REWIND"] = "true"
        self.doc = _Doc(ABS_ID)

    def tearDown(self):
        observation_trail.clear()
        if self._saved is None:
            os.environ.pop("SYNC_TRUST_CORROBORATED_REWIND", None)
        else:
            os.environ["SYNC_TRUST_CORROBORATED_REWIND"] = self._saved

    def _put(self, pct, device):
        """One external PUT, recorded the way the handler now records it."""
        kosync_server._record_external_kosync_observation(self.doc, pct, device[0], 1)

    def _corroborated(self, device):
        return kosync_server._backward_move_is_corroborated(self.doc, device[0], 1)

    def test_the_live_sequence_is_corroborated_by_the_second_report(self):
        """27% → 29% → 30% from the Kindle: proof exists from the second PUT on."""
        self._put(0.473, KOBO)

        self._put(0.27, KINDLE)
        self.assertFalse(self._corroborated(KINDLE), "one report is a stale reader being opened")

        self._put(0.29, KINDLE)
        self.assertTrue(self._corroborated(KINDLE), "read on from the new position — this is the rewind")

        self._put(0.30, KINDLE)
        self.assertTrue(self._corroborated(KINDLE))

    def test_opening_a_stale_reader_never_qualifies(self):
        """The case furthest-wins exists for: a device reports once and sits there."""
        self._put(0.473, KOBO)
        self._put(0.27, KINDLE)
        self._put(0.27, KINDLE)
        self._put(0.27, KINDLE)
        self.assertFalse(self._corroborated(KINDLE))

    def test_the_other_devices_reading_cannot_answer_for_this_one(self):
        """The Kobo advancing is not evidence that the Kindle's jump was deliberate."""
        self._put(0.27, KINDLE)
        self._put(0.473, KOBO)
        self._put(0.48, KOBO)
        self._put(0.49, KOBO)
        self.assertFalse(self._corroborated(KINDLE))
        self.assertTrue(self._corroborated(KOBO))

    def test_the_global_rewind_switch_still_governs_it(self):
        self._put(0.473, KOBO)
        self._put(0.27, KINDLE)
        self._put(0.29, KINDLE)
        os.environ["SYNC_TRUST_CORROBORATED_REWIND"] = "false"
        self.assertFalse(self._corroborated(KINDLE))

    def test_an_unlinked_document_has_no_book_to_judge_against(self):
        self.assertFalse(kosync_server._backward_move_is_corroborated(_Doc(None), KINDLE[0], 1))
        self.assertFalse(kosync_server._backward_move_is_corroborated(None, KINDLE[0], 1))

    def test_a_rejected_report_still_reaches_the_trail(self):
        """The ordering fix itself: recording must not depend on acceptance."""
        self._put(0.27, KINDLE)
        trail = observation_trail.get_trail("KoSync", ABS_ID, user_id=1)
        self.assertEqual([o.pct for o in trail], [0.27])
        self.assertEqual([o.device for o in trail], [KINDLE[0]])


class TestRecorderRunsBeforeTheGate(unittest.TestCase):
    """Guards the ordering in the handler source, which is the whole bug."""

    def test_observation_is_recorded_before_furthest_wins_can_return(self):
        import inspect

        source = inspect.getsource(kosync_server.kosync_put_progress)
        record_at = source.index("_record_external_kosync_observation")
        reject_at = source.index("Ignored progress from")
        self.assertLess(
            record_at, reject_at,
            "furthest-wins must not return before the report is observed — that is "
            "what made the corroborated-rewind rule unreachable for a second device",
        )


if __name__ == "__main__":
    unittest.main()
