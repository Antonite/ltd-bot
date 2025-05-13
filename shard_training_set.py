# shard_training_set.py  (fixed)
import pyarrow.parquet as pq, pyarrow as pa, numpy as np
import argparse, os
from tqdm import tqdm

def main(src, dst_dir, shards, seed):
    os.makedirs(dst_dir, exist_ok=True)
    writers = [None] * shards          # open on first write
    rng     = np.random.default_rng(seed)

    pf = pq.ParquetFile(src)
    for rg in tqdm(range(pf.num_row_groups)):
        tbl    = pf.read_row_group(rg)
        order  = rng.permutation(tbl.num_rows)
        shard_id = rng.integers(0, shards, len(order))

        for sid in range(shards):
            take = order[shard_id == sid]
            if take.size == 0:
                continue
            if writers[sid] is None:
                writers[sid] = pq.ParquetWriter(
                    os.path.join(dst_dir, f"{sid:03}.parquet"),
                    tbl.schema,               # ← schema here
                    compression="snappy",
                    use_dictionary=True,
                )
            writers[sid].write_table(tbl.take(pa.array(take)))

    for w in writers:
        if w is not None:
            w.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",     default="waveBuildAggregation.parquet")
    ap.add_argument("--dst_dir", default="shards")
    ap.add_argument("--shards",  type=int, default=64)
    ap.add_argument("--seed",    type=int, default=42)
    main(**vars(ap.parse_args()))
