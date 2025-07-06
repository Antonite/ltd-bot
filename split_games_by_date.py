#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from collections import defaultdict
from typing import Dict, List

import pymongo
from dateutil.parser import isoparse
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import BulkWriteError

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--uri",   default="mongodb://127.0.0.1:27017/", help="MongoDB URI")
    p.add_argument("--db",    default="legiontd2",                   help="Database name")
    p.add_argument("--source", default="games",                      help="Source collection")
    p.add_argument("--batch", type=int, default=1000,                help="Docs per bulk write")
    return p.parse_args()

# ---------- Bulk helper ----------
def flush_buffer(db: Database, coll_name: str, buffer: List[dict]) -> None:
    """Bulk-insert buffered docs and clear the buffer."""
    if not buffer:
        return
    coll: Collection = db[coll_name]

    try:
        coll.insert_many(buffer, ordered=False)
    except BulkWriteError as bwe:
        # Ignore duplicate-key errors so reruns don’t crash.
        if any(err["code"] != 11000 for err in bwe.details["writeErrors"]):
            raise
    buffer.clear()

# ---------- Main ----------
def main() -> None:
    args = parse_args()

    client = pymongo.MongoClient(args.uri)
    db: Database = client[args.db]
    src          = db[args.source]

    buffers: Dict[str, List[dict]] = defaultdict(list)
    total_processed = 0
    batch           = args.batch

    cursor = src.find({}, no_cursor_timeout=True)

    try:
        for doc in cursor:
            total_processed += 1

            raw_date = doc.get("date")
            if isinstance(raw_date, str):
                dt = isoparse(raw_date)                # handles trailing “Z”
            elif isinstance(raw_date, _dt.datetime):
                dt = raw_date
            else:
                continue                                # skip malformed

            coll_name = f"{args.source}_{dt:%Y_%m_%d}"
            buf = buffers[coll_name]
            buf.append(doc)

            if len(buf) >= batch:
                flush_buffer(db, coll_name, buf)

            if total_processed % 100_000 == 0:
                print(f"Processed {total_processed} docs…", file=sys.stderr)

    finally:
        cursor.close()

    # Final flush
    for coll_name, buf in buffers.items():
        flush_buffer(db, coll_name, buf)

    print(f"Done. Total documents processed: {total_processed}")

if __name__ == "__main__":
    main()
