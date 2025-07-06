#!/usr/bin/env python3
"""
aggregate_games.py — partition-aware, single-flush version.

Fix 2025-07-04-spell-no-extra-scan
---------------------------------
* Keeps every original comment and print statement.
* Reads from daily collections games_YYYY_MM_DD.
* Uses one in-RAM accumulator per day (flush at MEMORY_FLUSH_SIZE).
* Stores only _id (buildKey field removed).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Tuple, Dict, List, Optional

import pymongo
from shared import SPELLS, SKIP_SPELLS                            # NEW

EPS = 1e-6

# ───────────────────────────── Mongo config ─────────────────────────────
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["legiontd2"]

aggregate_collection = db["waveBuildAggregation"]
units_collection     = db["units"]
waves_collection     = db["waves"]

# ───────────────────────────── tunables ────────────────────────────────
BATCH_SIZE           = 10_000
CHUNK_SIZE           = 80_000
THREADS              = 24
MEMORY_FLUSH_SIZE    = 2_000_000      # flush when accumulator grows this big

PROJECTION: dict[str, int] = {
    "playersData.buildPerWave": 1,
    "playersData.leaksPerWave": 1,
    "playersData.mercenariesReceivedPerWave": 1,
    "playersData.endingWave": 1,
    "playersData.chosenChampionLocation": 1,
    "playersData.chosenSpell": 1,
    "playersData.chosenSpellLocation": 1,
}

# ─────────────────────────── helper utilities ──────────────────────────
def each_day(start_iso: str, end_iso: str) -> Iterable[datetime]:
    """Yield each UTC day between start and end (start inclusive, end exclusive)."""
    day = timedelta(days=1)
    s = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    e = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    while s < e:
        yield s
        s += day


def chunked_cursor(cursor, size: int = CHUNK_SIZE):
    """Yield chunks from a cursor so RAM stays low (< 8 GB for default settings)."""
    batch = []
    for d in cursor:
        batch.append(d)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_units_data() -> Dict[str, Dict[str, float]]:
    units: Dict[str, Dict[str, float]] = {}
    for doc in units_collection.find({}):
        raw = doc.get("goldBounty", 0)
        try:
            bounty = float(raw)
        except (TypeError, ValueError):
            bounty = 0.0

        units[doc["name"]] = {"bounty": bounty}

    return units


def load_waves_data() -> Dict[int, int]:
    return {int(d["levelNum"]): int(d.get("totalReward", 0)) for d in waves_collection.find({})}


def sum_merc_bounty(lst: list[str], units: dict) -> float:
    return sum((units[m]["bounty"] * 2 if m == "Imp" else units[m]["bounty"]) for m in lst if m in units)


def calc_leak_value(lst: list[str], units: dict) -> int:
    return sum(units[u]["bounty"] for u in lst if u in units)

# ─────────────────────── spell / key helpers ───────────────────────────
def _norm_coord(c: str) -> str:
    try:
        x, y = c.split("|")
        return f"{float(x):.1f}|{float(y):.1f}"
    except Exception:
        # keep original debug print
        print(f"exception in norm coord: {c}")
        return c.strip()


def build_key(
    wave: int,
    merc: float,
    build: list[str],
    spell_int: int,
    spell_loc: Optional[str],
) -> Tuple[str, bool]:
    """Return (aggregation-key, spell_matched)."""
    loc_norm = _norm_coord(spell_loc) if spell_loc else None
    spell_hit = False
    fighters: list[Tuple[str, float, float, int, int]] = []

    for token in build:
        unit, rest = token.split(":", 1)
        coords, stack = rest.rsplit(":", 1)
        x_s, y_s = coords.split("|")
        x, y, st = float(x_s), float(y_s), int(stack)

        flag = 0
        if wave >= 11 and spell_int and loc_norm and _norm_coord(coords) == loc_norm:
            flag = spell_int
            spell_hit = True

        fighters.append((unit, x, y, st, flag))

    fighters.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    fighters_part = ",".join(f"{u}:{x:.1f}:{y:.1f}:{s}:{f}" for u, x, y, s, f in fighters)
    return f"wave={wave}|merc={int(round(merc))}|{fighters_part}", spell_hit

# ───────────────────────────── core worker ─────────────────────────────
def process_games_chunk(args) -> Dict[str, list]:
    chunk, units, waves = args
    local: dict[str, list[int]] = {}

    for g in chunk:
        for p in g.get("playersData", []):
            # original guard - champion location
            if p.get("chosenChampionLocation", "-1|-1") != "-1|-1":
                continue

            bpw = p.get("buildPerWave", [])
            lpw = p.get("leaksPerWave", [])
            mpw = p.get("mercenariesReceivedPerWave", [])
            end_wave = int(p.get("endingWave", 0))

            spell_name   = p.get("chosenSpell", "")
            spell_loc    = p.get("chosenSpellLocation", "-1|-1")
            spell_int    = SPELLS.get(spell_name, 0)
            in_skip_list = spell_name in SKIP_SPELLS
            loc_valid    = spell_loc != "-1|-1"

            for idx, build in enumerate(bpw):
                wave = idx + 1
                if wave == end_wave or wave > 20:
                    continue

                merc = sum_merc_bounty(mpw[idx], units)
                if merc > 500:
                    continue
                leak_gold = calc_leak_value(lpw[idx], units)
                if leak_gold - merc > waves[idx + 1] + EPS: 
                    continue

                key, matched = build_key(wave, merc, build, spell_int, spell_loc if loc_valid else None)

                if wave >= 11:
                    if in_skip_list:
                        break
                    if loc_valid and not matched:
                        continue

                stats = local.setdefault(key, [0, 0])
                stats[0] += 1
                stats[1] += leak_gold

    return local


def bulk_upsert(batch: dict[str, list[int]]):
    if not batch:
        return
    reqs = [
        pymongo.UpdateOne(
            {"_id": k},
            {"$inc": {"totalOccurrences": v[0], "totalLeaked": v[1]}},
            upsert=True,
        )
        for k, v in batch.items()
    ]
    for i in range(0, len(reqs), BATCH_SIZE):
        aggregate_collection.bulk_write(reqs[i:i + BATCH_SIZE], ordered=False, bypass_document_validation=True)

# ───────────────────────────── driver ────────────────────────────────
def aggregate_wave_builds_daily(start_iso: str, end_iso: str):
    units = load_units_data()
    waves = load_waves_data()

    for day in each_day(start_iso, end_iso):
        coll_name = f"games_{day:%Y_%m_%d}"
        games_collection = db[coll_name]
        cnt = games_collection.estimated_document_count()
        print(f"\n▶ {coll_name} – {cnt} docs", flush=True)
        if not cnt:
            continue

        cursor = games_collection.find({}, PROJECTION).batch_size(CHUNK_SIZE)

        accumulator: dict[str, list[int]] = {}
        with ProcessPoolExecutor(max_workers=THREADS) as ex:
            for local in ex.map(
                process_games_chunk,
                ((c, units, waves) for c in chunked_cursor(cursor)),
                chunksize=1,
            ):
                for k, v in local.items():
                    stats = accumulator.setdefault(k, [0, 0])
                    stats[0] += v[0]
                    stats[1] += v[1]
                if len(accumulator) >= MEMORY_FLUSH_SIZE:
                    bulk_upsert(accumulator)
                    accumulator.clear()

        bulk_upsert(accumulator)   # final flush
        print("   done", flush=True)


# ──────────────────────────── CLI entry ──────────────────────────────
if __name__ == "__main__":
    start, end = (
        sys.argv[1:3]
        if len(sys.argv) == 3
        else ("2025-07-02:00:00Z", "2025-07-05T00:00:00Z")
    )
    aggregate_wave_builds_daily(start, end)
    print("Aggregation complete.")
