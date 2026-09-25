"""KoSync device arbiter — which of a book's device rows answers a GET.

A book linked through several KoSync hashes has one `kosync_user_progress` row per
hash, and the GET path has always reduced them with `max(percentage)`. That reduction
runs BEFORE leader selection ever sees "KoSync", so the disagreement between two
readers is resolved by the crudest rule in the system and the evidence that would
settle it — device identity, timestamps, provenance — is discarded on the way.

`max()` is also absorbing rather than merely blunt. A row is only ever written by an
external PUT on that exact hash; BookBridge's own writes advance `State` and never
touch the rows. So a device that once reported 86% keeps reporting 86% forever, and
keeps winning, until someone physically opens that device again. A reader working
through the same book on a second device is pulled forward on every GET, permanently.

This module applies the rule the rest of the bridge already uses for a backward move
(`SYNC_TRUST_CORROBORATED_REWIND`, "Honor a Deliberate Rewind"): a lower position wins
only when the reader PROVES it — same device, more than one report, advancing. One
reading is never enough, because one reading is exactly what a stale row looks like.
With no proof the furthest row still wins, so the protection against a stale second
reader dragging progress backwards is untouched.

Nothing here reads Flask or the database: it takes rows and answers with one of them,
which is what makes the policy testable on its own.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from src.services import observation_trail

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ON = "on"
_VALID_MODES = (MODE_OFF, MODE_SHADOW, MODE_ON)


def arbiter_mode() -> str:
    """Read per call so the Settings UI applies without a restart.

    Anything unrecognized reads as 'shadow' — the mode that logs but changes nothing —
    so a typo in the setting can never silently alter what a reader receives.
    """
    raw = str(os.environ.get("KOSYNC_ACTIVE_DEVICE_WINS", MODE_SHADOW) or MODE_SHADOW).strip().lower()
    if raw not in _VALID_MODES:
        logger.warning(
            "Invalid KOSYNC_ACTIVE_DEVICE_WINS=%r; using %s", raw, MODE_SHADOW,
        )
        return MODE_SHADOW
    return raw


@dataclass(frozen=True)
class DeviceChoice:
    """Which row answers this GET, and why."""
    row: object
    furthest_row: object
    reason: str
    active_device: str = ""
    # True when the arbiter picked something other than the furthest row. In shadow
    # mode the caller still returns `furthest_row`; this is what it logs.
    overrides_furthest: bool = False


def _pct(row) -> Optional[float]:
    try:
        value = getattr(row, "percentage", None)
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _device_name(row) -> str:
    return str(getattr(row, "device", "") or "").strip()


def choose_device_row(
    rows: Sequence[object],
    *,
    abs_id: str,
    user_id=None,
    parse_timestamp: Callable[[object], Optional[float]],
    is_internal_device: Optional[Callable[[Optional[str], Optional[str]], bool]] = None,
) -> Optional[DeviceChoice]:
    """Pick the row that should answer a GET, or None when there is nothing to pick.

    `parse_timestamp` and `is_internal_device` are injected rather than imported so
    this stays free of `kosync_server` (which imports this module).
    """
    candidates = [row for row in rows or [] if _pct(row) is not None]
    if not candidates:
        return None

    furthest = max(candidates, key=lambda row: _pct(row))
    furthest_pct = _pct(furthest)

    if arbiter_mode() == MODE_OFF:
        return DeviceChoice(furthest, furthest, "arbiter off — furthest wins")

    # Internal sync-bot rows are BookBridge's own writes wearing a device name. They
    # can never be the reader who is actively reading (failure mode #6).
    real = candidates
    if is_internal_device is not None:
        real = [
            row for row in candidates
            if not is_internal_device(getattr(row, "device", None), getattr(row, "device_id", None))
        ]

    devices = {_device_name(row) for row in real if _device_name(row)}
    if len(devices) < 2:
        return DeviceChoice(furthest, furthest, "one device — furthest wins")

    # Ask each device the same question the rewind gate asks: did this reader keep
    # reading? A device with no advancing sequence has proved nothing.
    active = []
    for device in sorted(devices):
        corroboration = observation_trail.evaluate(
            "KoSync", abs_id, user_id=user_id, device=device,
        )
        if corroboration.corroborated:
            active.append((device, corroboration))

    if not active:
        return DeviceChoice(furthest, furthest, "no device corroborated — furthest wins")
    if len(active) > 1:
        # Two readers genuinely moving at once is a real conflict, not something this
        # rule is entitled to settle. Leave it to furthest-wins.
        names = ",".join(device for device, _ in active)
        return DeviceChoice(furthest, furthest, f"{len(active)} devices active ({names}) — furthest wins")

    device, corroboration = active[0]
    chosen = max(
        (row for row in real if _device_name(row) == device),
        key=lambda row: _pct(row),
        default=None,
    )
    if chosen is None or chosen is furthest:
        return DeviceChoice(furthest, furthest, f"active device {device} already holds the furthest position")

    # Both timestamps were written by THIS server when each PUT arrived, so they share
    # one clock and comparing them is not the cross-service arbitration that leader
    # selection forbids (failure mode #7). A furthest row newer than the reader who is
    # actively reading is not the stale-sibling case this rule exists for.
    chosen_at = parse_timestamp(getattr(chosen, "timestamp", None))
    furthest_at = parse_timestamp(getattr(furthest, "timestamp", None))
    if chosen_at is None or furthest_at is None or furthest_at > chosen_at:
        return DeviceChoice(
            furthest, furthest,
            f"furthest position is not older than active device {device} — furthest wins",
        )

    return DeviceChoice(
        chosen,
        furthest,
        (
            f"active device {device} corroborated "
            f"({corroboration.advancing} advancing step(s)) at {_pct(chosen):.2%} "
            f"beats older furthest row at {furthest_pct:.2%}"
        ),
        active_device=device,
        overrides_furthest=True,
    )
