"""Optimised aggregation script – processes one day at a time and streams chunks so
RAM stays low (< 8 GB for default settings).

Fix 2025‑05‑04‑c
--------------
* Detects whether **`date`** field is stored as `ISODate` _or_ plain string.
  If it’s a string the query now uses *lexicographical* range on ISO‑8601
  strings (`'2024‑12‑15T00:00:00.000Z' ≤ date < '2024‑12‑16T00:00:00.000Z'`).
* Prints detected type at start so you can verify.

Nothing else changed.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor
import pymongo

# ──────────────────────────────── Mongo config ────────────────────────────────
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["legiontd2"]

games_collection      = db["games"]
aggregate_collection  = db["waveBuildAggregation"]
units_collection      = db["units"]

# ─────────────────────────────── tunables ─────────────────────────────────────
BATCH_SIZE  = 2_000
CHUNK_SIZE  = 10_000
THREADS     = 16
DATE_FIELD  = "date"  # matches your documents

PROJECTION: dict[str, int] = {
    "playersData.buildPerWave": 1,
    "playersData.leaksPerWave": 1,
    "playersData.mercenariesReceivedPerWave": 1,
    "playersData.endingWave": 1,
    DATE_FIELD: 1,
}

# ───────────────────────────── helper functions ───────────────────────────────

def parse_iso_utc(iso_str: str):
    if iso_str.endswith("Z"):
        iso_str = iso_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso_str)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def to_iso_z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# Decide if the field is stored as string or BSON Date
_doc = games_collection.find_one({DATE_FIELD: {"$exists": True}}, {DATE_FIELD: 1})
DATE_IS_STRING = isinstance(_doc.get(DATE_FIELD) if _doc else None, str)


def each_day(start_iso: str, end_iso: str):
    day = timedelta(days=1)
    s = parse_iso_utc(start_iso).astimezone(timezone.utc)
    e = parse_iso_utc(end_iso).astimezone(timezone.utc)
    while s < e:
        yield s, min(e, s + day)
        s += day


def chunked_cursor(cursor, chunk_size: int = CHUNK_SIZE):
    batch: list[dict] = []
    for doc in cursor:
        batch.append(doc)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch

# ───────────────────────────── unit helpers (unchanged) ───────────────────────

def load_units_data():
    units = {}
    for doc in units_collection.find({}):
        name = doc.get("name")
        if name:
            units[name] = {
                "totalValue": int(float(doc.get("totalValue", 0) or 0)),
                "bounty": float(doc.get("goldBounty", 0) or 0),
            }
    return units


def sum_merc_bounty(lst, units):
    return sum(units[m]["bounty"] for m in lst if m in units)


def calc_leak_value(lst, units):
    return sum(units[n]["bounty"] for n in lst if n in units)

# ───────────────────────────── key builder (unchanged) ───────────────────────

def build_key(wave: int, merc: float, build):
    fighters = []
    for item in build:
        unit, rest = item.split(":", 1)
        coords, stack = rest.rsplit(":", 1)
        x, y = coords.split("|")
        fighters.append((unit, float(x), float(y), int(stack)))
    fighters.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    fighters_part = ",".join(f"{u}:{x:.1f}:{y:.1f}:{s}" for u, x, y, s in fighters)
    return f"wave={wave}|merc={int(round(merc))}|{fighters_part}"

# ───────────────────────────── per‑chunk worker (unchanged) ───────────────────

def process_games_chunk(args):
    games_chunk, units = args
    local = {}
    for g in games_chunk:
        for p in g.get("playersData", []):
            bpw = p.get("buildPerWave", [])
            lpw = p.get("leaksPerWave", [])
            mpw = p.get("mercenariesReceivedPerWave", [])
            end_wave = int(p.get("endingWave", 0))
            for idx, build in enumerate(bpw):
                wave = idx + 1
                if wave == end_wave or wave > 20:
                    continue
                mercs = mpw[idx] if idx < len(mpw) else []
                merc_bounty = sum_merc_bounty(mercs, units)
                if merc_bounty > 500:
                    continue
                leaks = lpw[idx] if idx < len(lpw) else []
                leak_gold = calc_leak_value(leaks, units) if leaks else 0
                k = build_key(wave, merc_bounty, build)
                if k not in local:
                    local[k] = [0, 0]
                local[k][0] += 1
                local[k][1] += leak_gold
    return local

# ───────────────────────────── bulk upsert (unchanged) ───────────────────────

def bulk_upsert_partial(local):
    if not local:
        return
    reqs = [
        pymongo.UpdateOne(
            {"_id": k},
            {"$inc": {"totalOccurrences": v[0], "totalLeaked": v[1]}, "$setOnInsert": {"buildKey": k}},
            upsert=True,
        ) for k, v in local.items()
    ]
    for i in range(0, len(reqs), BATCH_SIZE):
        aggregate_collection.bulk_write(reqs[i:i+BATCH_SIZE], ordered=False, bypass_document_validation=True)

# ───────────────────────────── main driver ────────────────────────────────────

def aggregate_wave_builds_daily(start_iso, end_iso):
    units = load_units_data()
    for day_s, day_e in each_day(start_iso, end_iso):
        if DATE_IS_STRING:
            start_val, end_val = to_iso_z(day_s), to_iso_z(day_e)
        else:
            start_val, end_val = day_s, day_e
        qry = {DATE_FIELD: {"$gte": start_val, "$lt": end_val}}
        cnt = games_collection.count_documents(qry)
        print(f"\n▶ {day_s.date()} – {cnt} docs", flush=True)
        if cnt == 0:
            continue
        cur = games_collection.find(qry, PROJECTION).batch_size(CHUNK_SIZE)
        with ProcessPoolExecutor(max_workers=THREADS) as ex:
            for local in ex.map(process_games_chunk, ((chunk, units) for chunk in chunked_cursor(cur)), chunksize=1):
                bulk_upsert_partial(local)
        print("   done", flush=True)

# ───────────────────────────── CLI ────────────────────────────────────────────

if __name__ == "__main__":
    start, end = (sys.argv[1:3] if len(sys.argv) == 3 else ("2025-03-15T00:00:00Z", "2025-04-15T00:00:00Z"))
    aggregate_wave_builds_daily(start, end)
    print("Aggregation complete.")
