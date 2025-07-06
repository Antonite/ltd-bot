#!/usr/bin/env python3
"""
mongo_to_shards.py

Streams a MongoDB collection directly into sharded Parquet files in one pass
(100 000-row batches, 64 shards, Snappy compression, RNG seed 42).
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pymongo
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo_uri", default="mongodb://localhost:27017")
    ap.add_argument("--db", default="legiontd2")
    ap.add_argument("--coll", default="waveBuildAggregation")
    ap.add_argument("--dst_dir", default="shards")
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--batch", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total", type=int, default=45_200_000,
                    help="progress-bar total; omit to disable estimate")
    return ap.parse_args()


def open_writers(n: int) -> List[Optional[pq.ParquetWriter]]:
    return [None] * n


def write_batch(tbl: pa.Table, rng: np.random.Generator,
                writers: List[Optional[pq.ParquetWriter]], dst: str) -> None:
    order     = rng.permutation(tbl.num_rows)
    shard_ids = rng.integers(0, len(writers), len(order))

    for sid in range(len(writers)):
        mask = shard_ids == sid
        if not mask.any():
            continue

        writer = writers[sid]
        if writer is None:                       # first time for this shard
            path = os.path.join(dst, f"{sid:03}.parquet")
            writer = pq.ParquetWriter(path, tbl.schema,
                                      compression="snappy",
                                      use_dictionary=True)
            writers[sid] = writer

        # Narrow Optional → ParquetWriter for static checkers
        assert writer is not None
        writer.write_table(tbl.take(pa.array(order[mask])))


def main() -> None:
    args = parse_args()
    os.makedirs(args.dst_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    client = pymongo.MongoClient(args.mongo_uri, connect=False)
    col    = client[args.db][args.coll]

    cursor = col.find({}, {"_id": 1, "totalLeaked": 1, "totalOccurrences": 1},
                      sort=[("_id", 1)], batch_size=4096, no_cursor_timeout=True)

    writers = open_writers(args.shards)
    rng     = np.random.default_rng(args.seed)
    batch: list[dict] = []

    progress = tqdm(cursor, total=args.total, disable=args.total is None, smoothing=0.2)
    try:
        for doc in progress:
            batch.append(doc)
            if len(batch) == args.batch:
                tbl = pa.Table.from_pydict({
                    "_id":              [d["_id"]              for d in batch],
                    "totalLeaked":      [d["totalLeaked"]      for d in batch],
                    "totalOccurrences": [d["totalOccurrences"] for d in batch],
                })
                write_batch(tbl, rng, writers, args.dst_dir)
                batch.clear()

        if batch:  # final partial chunk
            tbl = pa.Table.from_pydict({
                "_id":              [d["_id"]              for d in batch],
                "totalLeaked":      [d["totalLeaked"]      for d in batch],
                "totalOccurrences": [d["totalOccurrences"] for d in batch],
            })
            write_batch(tbl, rng, writers, args.dst_dir)
    finally:
        for sid, w in enumerate(writers):
            if w is not None:
                logging.info("Closing shard %03d", sid)
                w.close()
        cursor.close()
        client.close()


if __name__ == "__main__":
    main()
