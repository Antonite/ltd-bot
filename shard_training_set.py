#!/usr/bin/env python3
"""
shard_training_set_v3.py  -  write shards whose *first* row-group is already
a random slice of the 50 M-row collection.
"""
from __future__ import annotations
import os, struct, tempfile, pickle, xxhash, pymongo, pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import Dict, List, Tuple

# ── parameters ──────────────────────────────────────────────────────────
SEED       = 0x1234_5678_9ABC_DEF0        # change = new shuffle
SHARDS     = 64
ROW_GROUP  = 10_000                       # rows per row-group
DST_DIR    = "shards"

# ── helper: open writers lazily, return concrete type ───────────────────
def get_writer(sid: int, schema: pa.Schema,
               writers: Dict[int, pq.ParquetWriter]) -> pq.ParquetWriter:
    w = writers.get(sid)
    if w is None:
        path = os.path.join(DST_DIR, f"{sid:03}.parquet")
        w = pq.ParquetWriter(path, schema,
                             compression="snappy", use_dictionary=True)
        writers[sid] = w
    return w

# ── 1️⃣  stream collection, assign a 64-bit random key, keep on disk ──
tmp = tempfile.NamedTemporaryFile(delete=False).name
with open(tmp, "wb") as fh, \
     pymongo.MongoClient("mongodb://localhost:27017", connect=False) as cli:

    cur = cli.legiontd2.waveBuildAggregation.find(
        {}, {"_id":1,"totalLeaked":1,"totalOccurrences":1},
        batch_size=4096, no_cursor_timeout=True)

    for doc in tqdm(cur, desc="hash-writing"):
        key  = xxhash.xxh64(doc["_id"], seed=SEED).intdigest()
        blob = pickle.dumps(doc, protocol=-1)
        fh.write(struct.pack("<Q", key))
        fh.write(struct.pack("<I", len(blob)))
        fh.write(blob)
    cur.close()

# ── 2️⃣  load, sort by key, flush round-robin into shards ───────────────
def iter_pairs(path: str):
    with open(path, "rb") as f:
        while hdr := f.read(12):
            key, n = struct.unpack("<QI", hdr)
            yield key, pickle.loads(f.read(n))

pairs: List[Tuple[int, dict]] = list(iter_pairs(tmp))
os.unlink(tmp)                         # remove temp file
pairs.sort(key=lambda t: t[0])         # in-place 64-bit sort

writers: Dict[int, pq.ParquetWriter] = {}
buffers: List[List[dict]] = [[] for _ in range(SHARDS)]

for _, doc in tqdm(pairs, desc="writing shards"):
    sid = doc["_id"].__hash__() % SHARDS         # deterministic, cheap
    buf = buffers[sid]
    buf.append(doc)

    if len(buf) >= ROW_GROUP:
        tbl = pa.Table.from_pydict({
            "_id":              [d["_id"]              for d in buf],
            "totalLeaked":      [d["totalLeaked"]      for d in buf],
            "totalOccurrences": [d["totalOccurrences"] for d in buf],
        })
        get_writer(sid, tbl.schema, writers).write_table(tbl)
        buf.clear()

# final flush for partially filled buffers
for sid, buf in enumerate(buffers):
    if buf:
        tbl = pa.Table.from_pydict({
            "_id":              [d["_id"]              for d in buf],
            "totalLeaked":      [d["totalLeaked"]      for d in buf],
            "totalOccurrences": [d["totalOccurrences"] for d in buf],
        })
        get_writer(sid, tbl.schema, writers).write_table(tbl)

for w in writers.values():
    w.close()
