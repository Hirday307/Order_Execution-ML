import pytest

from adaptive_exec.loaders import lobster_to_events
from adaptive_exec.synthetic import generate_lobster_day


@pytest.fixture(scope="session")
def raw_lobster():
    """20 minutes of synthetic LOBSTER messages + book, 09:30-09:50."""
    return generate_lobster_day(seed=7, start_s=34_200.0, end_s=35_400.0, rate_per_s=8.0)


@pytest.fixture(scope="session")
def events(raw_lobster):
    msg, book = raw_lobster
    return lobster_to_events(msg, book, "2012-06-21", levels=10)
