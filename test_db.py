import argparse
import re
import sys
from collections import Counter

import pymongo

# ────────────────────────────── DB handles ───────────────────────
cli   = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db    = cli["legiontd2"]
agg   = db["waveBuildAggregation"]
waves = db["waves"]

# pre-load wave → reward (gold bounty) mapping
WAVE_VAL = {
    int(d["levelNum"]): float(d.get("totalReward", 0.0))
    for d in waves.find({}, {"levelNum": 1, "totalReward": 1})
}
if not WAVE_VAL:
    sys.exit("✖ waves collection is empty – cannot compute threat values")

# regex to pull wave & merc out of buildKey
KEY_RE = re.compile(r"wave=(\d+)\|merc=(\d+)\|")

# ────────────────────────────── scan ─────────────────────────────
bad, checked = [], 0
for doc in agg.find({}, {"buildKey": 1, "totalLeaked": 1, "totalOccurrences": 1}):
    m = KEY_RE.match(doc["buildKey"])
    if not m:
        continue                        # malformed key – skip / log as needed
    wave_num      = int(m.group(1))
    merc_bounty   = float(m.group(2))
    threat        = WAVE_VAL.get(wave_num, 0.0) + merc_bounty
    if threat == 0:
        continue                        # no threat info – skip

    frac = (doc["totalLeaked"] / max(doc["totalOccurrences"], 1)) / threat
    if frac > 1.0001:                   # 0.01 % tolerance
        bad.append({
            "_id": doc["_id"],
            "wave": wave_num,
            "merc": merc_bounty,
            "frac": round(frac, 4),
            "leaked": doc["totalLeaked"],
            "occ": doc["totalOccurrences"],
            "threat": threat,
        })
    checked += 1
    if checked % 10000 == 0:
        print(f"✓ scanned {checked:,} documents – {len(bad)} inconsistenc{'y' if len(bad)==1 else 'ies'} found\n")

# ────────────────────────────── summary ──────────────────────────
print(f"✓ scanned {checked:,} documents – {len(bad)} inconsistenc{'y' if len(bad)==1 else 'ies'} found\n")

for entry in bad[:50]:                  # show first 50 only
    print(f"{entry['_id']} | wave={entry['wave']:>2} "
          f"merc={entry['merc']:>4} | "
          f"leak/occ={entry['leaked']}/{entry['occ']} "
          f"(frac={entry['frac']}) > threat={entry['threat']}")
if len(bad) > 50:
    print(f"... and {len(bad) - 50} more")
sys.exit(1 if bad else 0)
