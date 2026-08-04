from datetime import date, datetime, timezone

from backend.api.routes import calls


def test_completed_at_is_bucketed_by_ist_day_not_utc():
    """A call at 02:00 IST belongs to that IST day.

    Stored as UTC it is 20:30 the *previous* day, so bucketing on the raw UTC
    date put it under "Yesterday" — which reads to a recruiter as their calls
    silently moving between days.
    """
    # 2026-08-04 02:00 IST == 2026-08-03 20:30 UTC
    completed_utc = datetime(2026, 8, 3, 20, 30)
    assert calls.ist_date_of(completed_utc) == date(2026, 8, 4)
    # The naive .date() this replaced would have said the 3rd.
    assert completed_utc.date() == date(2026, 8, 3)


def test_late_evening_ist_call_stays_on_its_own_day():
    """The mirror case: 23:00 IST is 17:30 UTC the same day — must not slide
    forward either."""
    completed_utc = datetime(2026, 8, 3, 17, 30)
    assert calls.ist_date_of(completed_utc) == date(2026, 8, 3)


def test_slicer_matches_an_early_morning_call_under_today():
    """The cache path must agree with the SQL path. Bounds are IST dates."""
    todays_bounds = (date(2026, 8, 4), date(2026, 8, 4))
    early_morning = {"completed_at": datetime(2026, 8, 3, 20, 30), "outcome": "Not Connected"}

    assert calls.call_matches_slicer(
        early_morning, todays_bounds, None, use_completed_at=True
    ) is True

    yesterdays_bounds = (date(2026, 8, 3), date(2026, 8, 3))
    assert calls.call_matches_slicer(
        early_morning, yesterdays_bounds, None, use_completed_at=True
    ) is False


def test_ist_today_is_ahead_of_utc_during_the_gap():
    """Between 00:00 and 05:30 IST the two calendars disagree; ist_today must
    follow the recruiter, not the server."""
    assert calls.ist_today() == datetime.now(timezone.utc).astimezone(calls.IST).date()


def test_range_sql_no_longer_uses_bare_current_date():
    for rng in ("today", "yesterday"):
        frag, params = build = calls.build_range_sql(
            rng, None, None, use_completed_at=True, col_prefix="c."
        )
        assert "Asia/Kolkata" in frag, rng
        # A bare CURRENT_DATE would reintroduce the UTC rollover.
        assert "CURRENT_DATE" not in frag, rng
        assert params == []


def test_due_date_ranges_also_use_ist():
    frag, _ = calls.build_range_sql("today", None, None, use_completed_at=False, col_prefix="c.")
    assert "Asia/Kolkata" in frag
    assert "CURRENT_DATE" not in frag


def test_custom_range_still_takes_explicit_dates():
    frag, params = calls.build_range_sql(
        "custom", date(2026, 7, 1), date(2026, 7, 31), use_completed_at=True, col_prefix="c."
    )
    assert params == ["2026-07-01", "2026-07-31"]
    # Explicit dates are still interpreted as IST days, matching what the user
    # picked in the date box.
    assert "Asia/Kolkata" in frag
