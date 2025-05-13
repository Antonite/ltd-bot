import requests
import pymongo
import datetime
import os

API_KEY = os.environ["LTD2_API_KEY"]
HEADERS = {"x-api-key": API_KEY}

# Mongo connection (local)
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["legiontd2"]  

# Collections
games_collection = db["games"]
units_collection = db["units"]
waves_collection = db["waves"]
fetch_state_collection = db["fetchState"]

DAILY_LIMIT = 1000

def fetch_units(version=None, limit=300):
    """
    Fetch multiple units for a given version from /units/byVersion/{version}.
    If version is not specified, you can pass the most recent version from the API.
    'limit=300' fetches all units in one go.
    """

    base_url = f"https://apiv2.legiontd2.com/units/byVersion/{version}"
    params = {
        "enabled": "true",  # Set to False if you also want disabled units
        "limit": limit
    }
    resp = requests.get(base_url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        print("Error fetching units:", resp.status_code)
        return

    unit_list = resp.json()
    for unit_doc in unit_list:
        # Construct a unique _id, e.g. "unitId_version"
        # The API might not give a simple 'id' for units, so you can combine name + version or sortHelper + version
        name = unit_doc.get("name", "unknown")
        doc_id = f"{name}_{version}"  # or use sortHelper
        unit_doc.pop("_id", None)
        units_collection.update_one(
            {"_id": doc_id},
            {"$set": unit_doc},
            upsert=True
        )
    print(f"Fetched & upserted {len(unit_list)} units for version '{version}'.")

def fetch_waves(offset=0, limit=50):
    """
    Fetch waves in batches (50 is the max recommended) from /info/waves/{offset}/{limit}.
    Keep incrementing offset until no more.
    """
    while True:
        base_url = f"https://apiv2.legiontd2.com/info/waves/{offset}/{limit}"
        resp = requests.get(base_url, headers=HEADERS)
        if resp.status_code != 200:
            print("Error fetching waves:", resp.status_code)
            break

        wave_list = resp.json()
        if not wave_list:
            break
        
        # Upsert each wave by its 'id'
        for wave_doc in wave_list:
            wave_id = wave_doc.get("_id")
            if wave_id:
                wave_doc.pop("_id", None)
                waves_collection.update_one(
                    {"_id": wave_id},
                    {"$set": wave_doc},
                    upsert=True
                )
        print(f"Fetched & upserted {len(wave_list)} waves (offset={offset}).")

        if len(wave_list) < limit:
            break
        offset += limit

def get_fetch_state():
    """
    Retrieves current fetch state from the fetchState collection.
    If none exists, create a default (yesterday, offset=0, etc.).
    """
    state = fetch_state_collection.find_one({"_id": "games_fetch"})
    if state is None:
        # default to starting from yesterday
        yesterday = datetime.datetime.now(datetime.UTC).date() - datetime.timedelta(days=1)
        state = {
            "_id": "games_fetch",
            "currentDate": str(yesterday),  # store as string "YYYY-MM-DD"
            "currentOffset": 0,
            "requestsUsedToday": 0,
            "dailyLimit": DAILY_LIMIT
        }
        fetch_state_collection.insert_one(state)
    return state

def save_fetch_state(state):
    """
    Save the updated fetch state doc back to Mongo.
    """
    fetch_state_collection.update_one(
        {"_id": "games_fetch"},
        {"$set": state},
        upsert=True
    )

def fetch_one_day(date_str, start_offset, requests_used, daily_limit):
    """
    Fetches up to one day of games from date_str (e.g. "2023-07-04"),
    starting at offset = start_offset, returning:
      - new_offset: where we ended
      - requests_used: how many requests were used
      - done_for_day: bool indicating if we fetched all games for that day
    We’ll stop if we exceed the daily limit or if no more data is returned.
    """
    # Build date range
    # For example, 00:00:00 up to 23:59:59 that same day
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    date_after = date_obj.strftime("%Y-%m-%d 00:00:00")
    # Next day => date_obj + 1 day => we do second's minus 1 or something:
    date_before_obj = date_obj + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
    date_before = date_before_obj.strftime("%Y-%m-%d %H:%M:%S")

    offset = start_offset
    limit = 50
    
    base_url = "https://apiv2.legiontd2.com/games"
    
    while True:
        # Check if we can make another request within our daily limit
        if requests_used >= daily_limit:
            print("Hit daily limit - stopping.")
            return offset, requests_used, False
        
        # Build params
        params = {
            "limit": limit,
            "offset": offset,
            "dateAfter": date_after,
            "dateBefore": date_before,
            "includeDetails": "true",
            "sortBy": "date",
            "sortDirection": 1,
            "queueType": ["Normal"]  # or other queues
        }
        
        resp = requests.get(base_url, headers=HEADERS, params=params)
        requests_used += 1
        if resp.status_code == 404:
            # No more data => we are done with this day
            break
        if resp.status_code != 200:
            print("Error fetching games:", resp.status_code)
            # API error. likely out of API requests
            return offset, requests_used, False
        
        data = resp.json()
        
        # Upsert each match by its unique game '_id'
        for game_doc in data:
            game_id = game_doc.get("_id")
            if game_id:
                # remove the _id from doc so we can upsert
                game_doc.pop("_id", None)
                if int(game_doc.get("endingWave")) < 2:
                    continue
                games_collection.update_one(
                    {"_id": game_id},
                    {"$set": game_doc},
                    upsert=True
                )
        
        print(f"Fetched & upserted {len(data)} games (offset={offset}) on {date_str}.")

        # If we got fewer than limit, that means no more data remains
        if len(data) < limit:
            offset += len(data)
            break
        else:
            offset += limit
    
    # If we exit normally, that means we’re done with this day
    return offset, requests_used, True

def main():
    state = get_fetch_state()
    print("Loaded fetch state:", state)

    current_date_str = state["currentDate"]
    current_offset = state["currentOffset"]
    requests_used = state["requestsUsedToday"]
    daily_limit = state["dailyLimit"]

    if requests_used >= daily_limit:
        requests_used = 0
    
    # We’ll do a loop that tries to fetch day by day going backwards
    while True:
        print(f"Fetching day {current_date_str} from offset {current_offset}, requests used {requests_used}/{daily_limit}")
        new_offset, requests_used, done_for_day = fetch_one_day(
            current_date_str, current_offset, requests_used, daily_limit
        )

        print(f"requests used updated {requests_used}")

        current_offset = new_offset

        # If we are done with that day (no more data or partial fetch that ended), move to previous day
        if done_for_day:
            # Move to previous day
            day_obj = datetime.datetime.strptime(current_date_str, "%Y-%m-%d")
            prev_day_obj = day_obj - datetime.timedelta(days=1)
            prev_day_str = prev_day_obj.strftime("%Y-%m-%d")
            
            current_date_str = prev_day_str
            current_offset = 0
            
            # We also might reset requestsUsed if you want to handle each day’s limit separately,
            # but presumably your daily limit resets at real midnight, not per game day. So we keep incrementing.
            
            # store new date & offset
            state["currentDate"] = current_date_str
            state["currentOffset"] = 0
            save_fetch_state(state)
        else:
            # Save partial progress
            state["currentOffset"] = current_offset
            save_fetch_state(state)
        
        # If we ran out of daily requests, we stop
        if requests_used >= daily_limit:
            print("Reached daily limit. Storing state, exiting.")
            break

    print("Data fetching complete.")


if __name__ == "__main__":
    main()