"""
Manual overrides for per‑unit stack limits and extra fighter IDs
that are missing from MongoDB.

•  Put every unit that needs a non‑default stack limit in
   STACK_LIMIT_OVERRIDES — units not listed default to 0.

•  Add any fighters that do not exist in the DB to EXTRA_FIGHTERS.
"""

# unitId ➜ max‑stack
STACK_LIMIT_OVERRIDES: dict[str, int] = {
    "kingpin_unit_id": 225,
    "nekomata_unit_id": 7,
    "infiltrator_unit_id": 4,
    "orchid_unit_id": 12,
    "sakura_unit_id": 40,
}

# fighters missing in Mongo
EXTRA_FIGHTERS: list[str] = [
    "hell_raiser_buffed_unit_id",
    "pack_rat_nest_unit_id",
]
