import requests
import pymongo
import datetime
from zoneinfo import ZoneInfo  # Python ≥3.9
import os

API_KEY = os.environ["LTD2_API_KEY"]
HEADERS = {"x-api-key": API_KEY}

# Mongo connection (local)
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["legiontd2"]

# Collections (non-game collections stay the same)
units_collection = db["units"]
waves_collection = db["waves"]
spells_collection = db["spells"]

DAILY_LIMIT = 1000  # still respected inside fetch_one_day


# ---------- non-game helpers (unchanged) ----------

def fetch_spells():
    base_url = "https://apiv2.legiontd2.com/info/spells/0/50"
    params = {"enabled": "true"}
    resp = requests.get(base_url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        print("Error fetching spells:", resp.status_code)
        return

    spell_list = resp.json()
    for spell_doc in spell_list:
        _id = spell_doc.pop("_id", "unknown")
        spells_collection.update_one({"_id": _id}, {"$set": spell_doc}, upsert=True)
    print(f"Fetched & upserted {len(spell_list)} spells.")


def fetch_units(version=None, limit=300):
    base_url = f"https://apiv2.legiontd2.com/units/byVersion/{version}"
    params = {"enabled": "true", "limit": limit}
    resp = requests.get(base_url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        print("Error fetching units:", resp.status_code)
        return

    unit_list = resp.json()
    for unit_doc in unit_list:
        name = unit_doc.get("name", "unknown")
        doc_id = f"{name}_{version}"
        unit_doc.pop("_id", None)
        units_collection.update_one({"_id": doc_id}, {"$set": unit_doc}, upsert=True)
    print(f"Fetched & upserted {len(unit_list)} units for version '{version}'.")


def fetch_waves(offset=0, limit=50):
    while True:
        base_url = f"https://apiv2.legiontd2.com/info/waves/{offset}/{limit}"
        resp = requests.get(base_url, headers=HEADERS)
        if resp.status_code != 200:
            print("Error fetching waves:", resp.status_code)
            break

        wave_list = resp.json()
        if not wave_list:
            break

        for wave_doc in wave_list:
            wave_id = wave_doc.pop("_id", None)
            if wave_id:
                waves_collection.update_one({"_id": wave_id}, {"$set": wave_doc}, upsert=True)
        print(f"Fetched & upserted {len(wave_list)} waves (offset={offset}).")

        if len(wave_list) < limit:
            break
        offset += limit


# ---------- game fetch ----------

def fetch_one_day(date_str, start_offset, requests_used, daily_limit, games_collection):
    """
    Download every game on a single calendar day.
    """
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    # 00:00 of target day  ≤  game.date  <  00:00 of next day
    date_after = date_obj.strftime("%Y-%m-%d 00:00:00")
    date_before = (date_obj + datetime.timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")

    offset, limit = start_offset, 50
    base_url = "https://apiv2.legiontd2.com/games"

    while True:
        if requests_used >= daily_limit:
            print("Hit daily limit – stopping.")
            return requests_used

        params = {
            "limit": limit,
            "offset": offset,
            "dateAfter": date_after,
            "dateBefore": date_before,
            "includeDetails": "true",
            "sortBy": "date",
            "sortDirection": 1,
            "queueType": ["Normal"],
        }

        resp = requests.get(base_url, headers=HEADERS, params=params)
        requests_used += 1
        if resp.status_code == 404:          # no more data
            break
        if resp.status_code != 200:
            print("Error fetching games:", resp.status_code)
            break

        data = resp.json()
        for game_doc in data:
            game_id = game_doc.get("_id")
            if not game_id:
                continue
            if int(game_doc.get("endingWave", 0)) < 2:
                continue
            game_doc.pop("_id", None)
            games_collection.update_one({"_id": game_id}, {"$set": game_doc}, upsert=True)

        print(f"Fetched & upserted {len(data)} games (offset={offset}) on {date_str}.")

        if len(data) < limit:
            break
        offset += limit

    return requests_used


# ---------- entry-point ----------

def main():
    # Local timezone – adjust if needed
    eastern = ZoneInfo("America/New_York")

    now_et = datetime.datetime.now(eastern)
    yesterday = (now_et - datetime.timedelta(days=1)).date()
    date_str = yesterday.strftime("%Y-%m-%d")

    collection_name = f"games_{yesterday.strftime('%Y_%m_%d')}"
    games_collection = db[collection_name]

    print(f"Starting fetch for {date_str}; results will go to collection '{collection_name}'.")

    fetch_one_day(
        date_str=date_str,
        start_offset=0,
        requests_used=0,
        daily_limit=DAILY_LIMIT,
        games_collection=games_collection,
    )

    print("Finished fetching yesterday's games.")


if __name__ == "__main__":
    # fetch_units("12.05.3")
    # fetch_spells()
    main()
