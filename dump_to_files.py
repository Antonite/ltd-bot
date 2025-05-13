# dump_wave.py
import pymongo, pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm

MONGO_URI = "mongodb://localhost:27017"
DB, COLL  = "legiontd2", "waveBuildAggregation"
OUT_FILE  = "waveBuildAggregation.parquet"

client = pymongo.MongoClient(MONGO_URI, connect=False)
col    = client[DB][COLL]

cursor = col.find({},  # all docs
                  {"_id": 1, "totalLeaked": 1, "totalOccurrences": 1},
                  sort=[("_id", 1)],
                  batch_size=4096,
                  no_cursor_timeout=True)

writer = None
batch  = []
BATCH  = 100_000

for doc in tqdm(cursor, total=41_531_238):         # adjust total if needed
    batch.append(doc)
    if len(batch) == BATCH:
        tbl = pa.Table.from_pydict({
            "_id":              [d["_id"]              for d in batch],
            "totalLeaked":      [d["totalLeaked"]      for d in batch],
            "totalOccurrences": [d["totalOccurrences"] for d in batch],
        })
        if writer is None:
            writer = pq.ParquetWriter(OUT_FILE, tbl.schema, compression="snappy")
        writer.write_table(tbl)
        batch.clear()

if batch:
    writer.write_table(pa.Table.from_pydict({
        "_id":              [d["_id"]              for d in batch],
        "totalLeaked":      [d["totalLeaked"]      for d in batch],
        "totalOccurrences": [d["totalOccurrences"] for d in batch],
    }))
writer.close()
