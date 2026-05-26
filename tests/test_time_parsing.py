from datetime import datetime

from core.utils import parse_chetime


def test_parse_chetime_invalid_and_rollover() -> None:
    ref = datetime(2026, 3, 2, 0, 0, 10)

    # Invalid format -> returns reference (seconds preserved, microseconds dropped)
    out = parse_chetime("bad", reference=ref)
    assert out == ref.replace(microsecond=0)

    # Rollover forward: chetime slightly ahead but reference near midnight should keep same day
    out2 = parse_chetime("000020", reference=ref)
    assert out2.date() == ref.date()
    assert out2.hour == 0 and out2.minute == 0 and out2.second == 20

    # Rollover backward: reference just after midnight but chetime late previous day -> should go to previous day
    out3 = parse_chetime("235950", reference=ref)
    assert out3 < ref
