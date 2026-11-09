"""
src/inference/calendar.py

FX trading calendar and gap classification module for real-time M15 inference.
Extends and operationalizes Phase 1 quality checks.

Classifies incoming observation intervals into:
- Normal (delta == 15 min)
- Active-session missing bars (delta in 30..60 min during active hours)
- Market closures (Weekends, Christmas, New Year)
- Uncertain / Irregular gaps (non-monotonic, non-multiples of 15 min, unclassified gaps)
"""

from dataclasses import dataclass
from typing import Tuple
import pandas as pd


class UncertainGapError(ValueError):
    """Raised when a time gap cannot be classified reliably by the calendar policy."""

    pass


def is_fx_holiday(dt: pd.Timestamp) -> bool:
    """Check if date falls on standard international FX holiday closures."""
    # Christmas: Dec 24 after 21:00 UTC through Dec 25
    if dt.month == 12 and dt.day == 25:
        return True
    if dt.month == 12 and dt.day == 24 and dt.hour >= 21:
        return True
    # New Year: Dec 31 after 21:00 UTC through Jan 1
    if dt.month == 1 and dt.day == 1:
        return True
    if dt.month == 12 and dt.day == 31 and dt.hour >= 21:
        return True
    return False


@dataclass(frozen=True)
class GapClassification:
    """Classification result for a time interval between two observations."""

    gap_type: str  # 'normal', 'active_missing', 'market_closure', 'uncertain'
    n_transitions: int  # Number of 15-minute Markov steps n (for active session)
    missing_bars: int  # Number of unobserved intermediate bars (n - 1)
    description: str  # Human-readable description
    requires_reset: bool  # True if gap policy mandates prior reset to startprob_


class FXCalendarGapHandler:
    """
    Evaluates timestamp deltas and applies FX market calendar rules to classify intervals.
    """

    def __init__(self, max_active_gap_minutes: int = 300):
        """
        Parameters
        ----------
        max_active_gap_minutes : int
            Maximum duration (in minutes) during open market hours to treat as
            missing bars and propagate via A^n. Gaps exceeding this during active
            sessions are treated as anomalous / uncertain (default: 300 minutes, 5 hours).
        """
        self.max_active_gap_minutes = max_active_gap_minutes

    def classify(
        self,
        t_prev: pd.Timestamp,
        t_curr: pd.Timestamp,
        strict: bool = True,
    ) -> GapClassification:
        """
        Classify the interval between consecutive observation timestamps.

        Parameters
        ----------
        t_prev : pd.Timestamp
            Timestamp of previous observation (must be UTC).
        t_curr : pd.Timestamp
            Timestamp of current incoming observation (must be UTC).
        strict : bool
            If True, raises UncertainGapError on ambiguous or irregular gaps.

        Returns
        -------
        GapClassification
        """
        t_prev = pd.to_datetime(t_prev)
        t_curr = pd.to_datetime(t_curr)

        # Enforce UTC timezone awareness or consistent comparison
        if t_prev.tz is not None and t_curr.tz is None:
            t_curr = t_curr.tz_localize(t_prev.tz)
        elif t_prev.tz is None and t_curr.tz is not None:
            t_prev = t_prev.tz_localize(t_curr.tz)

        delta = t_curr - t_prev
        delta_seconds = delta.total_seconds()

        # 1. Non-monotonic or duplicate check
        if delta_seconds <= 0:
            msg = f"Non-monotonic or duplicate timestamp: t_prev={t_prev}, t_curr={t_curr} (delta={delta_seconds}s)"
            if strict:
                raise UncertainGapError(msg)
            return GapClassification(
                gap_type="uncertain",
                n_transitions=0,
                missing_bars=0,
                description=msg,
                requires_reset=False,
            )

        # 2. Check for irregular non-multiples of 15 minutes (allowing 5-second jitter)
        remainder_seconds = delta_seconds % 900.0
        is_exact_multiple = (remainder_seconds < 5.0) or (remainder_seconds > 895.0)
        if not is_exact_multiple:
            msg = f"Irregular interval not a multiple of 15 minutes: {delta_seconds} seconds ({t_prev} -> {t_curr})"
            if strict:
                raise UncertainGapError(msg)
            return GapClassification(
                gap_type="uncertain",
                n_transitions=int(round(delta_seconds / 900.0)),
                missing_bars=max(0, int(round(delta_seconds / 900.0)) - 1),
                description=msg,
                requires_reset=False,
            )

        n_steps = int(round(delta_seconds / 900.0))
        missing_bars = n_steps - 1

        # 3. Normal consecutive bar (15 minutes, missing_bars == 0)
        if missing_bars == 0:
            return GapClassification(
                gap_type="normal",
                n_transitions=1,
                missing_bars=0,
                description="Normal consecutive 15-minute bar",
                requires_reset=False,
            )

        # 4. Check Weekend Closure
        # Friday after 18:00 UTC through Sunday 21:00 UTC or Monday morning
        is_weekend = False
        prev_wday = t_prev.weekday()  # Monday is 0, Friday is 4, Saturday 5, Sunday 6
        curr_wday = t_curr.weekday()

        if prev_wday == 4 and t_prev.hour >= 18 and (curr_wday in [6, 0]):
            is_weekend = True
        elif prev_wday == 4 and curr_wday == 6:
            is_weekend = True
        elif prev_wday in [4, 5] and curr_wday in [6, 0]:
            is_weekend = True
        elif prev_wday == 5 or curr_wday == 5:
            is_weekend = True

        if is_weekend:
            return GapClassification(
                gap_type="market_closure",
                n_transitions=n_steps,
                missing_bars=missing_bars,
                description=f"Weekend closure ({t_prev.strftime('%a %H:%M')} -> {t_curr.strftime('%a %H:%M')})",
                requires_reset=True,
            )

        # 5. Check Holiday Closure
        if is_fx_holiday(t_prev) or is_fx_holiday(t_curr):
            return GapClassification(
                gap_type="market_closure",
                n_transitions=n_steps,
                missing_bars=missing_bars,
                description=f"Holiday market closure around {t_prev.strftime('%Y-%m-%d')}",
                requires_reset=True,
            )

        # 6. Active Session Missing Bars
        gap_minutes = int(round(delta_seconds / 60.0))
        if gap_minutes <= self.max_active_gap_minutes:
            return GapClassification(
                gap_type="active_missing",
                n_transitions=n_steps,
                missing_bars=missing_bars,
                description=f"Active session gap of {missing_bars} missing bar(s) ({gap_minutes} min)",
                requires_reset=False,
            )

        # 7. Unclassified long gap during active trading hours (anomaly)
        msg = f"Unclassified long gap of {gap_minutes} minutes during weekday trading ({t_prev} -> {t_curr})"
        if strict:
            raise UncertainGapError(msg)
        return GapClassification(
            gap_type="uncertain",
            n_transitions=n_steps,
            missing_bars=missing_bars,
            description=msg,
            requires_reset=True,
        )
