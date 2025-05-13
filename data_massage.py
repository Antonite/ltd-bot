import pymongo

client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["legiontd2"]
games_col = db["games"]

# Delete any games whose 'endingWave' is less than 2
result = games_col.delete_many({"endingWave": {"$lt": 2}})
print(f"Deleted {result.deleted_count} documents where endingWave < 2")