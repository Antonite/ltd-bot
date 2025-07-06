#!/usr/bin/env python3
"""
run_yesterday.py - fetch yesterday's games **and** aggregate them in one command.

Usage:
    python run_yesterday.py
"""

from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

import retrieve              # (your existing retrieve.py)
import aggregate_games       # (your existing aggregate_games.py)


def main() -> None:
    # Determine "yesterday" in the same timezone used by retrieve.py
    eastern   = ZoneInfo("America/New_York")
    yesterday = (datetime.now(eastern) - timedelta(days=1)).date()

    # 1) ── Fetch yesterday’s games into collection  games_YYYY_MM_DD
    retrieve.main()

    # 2) ── Aggregate that single day’s collection
    start_iso = yesterday.strftime("%Y-%m-%dT00:00:00Z")
    end_iso   = (yesterday + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    aggregate_games.aggregate_wave_builds_daily(start_iso, end_iso)


if __name__ == "__main__":
    main()
