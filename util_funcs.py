import h5py
import json, pymongo

config = None
def read_config(path="config.json"):
    global config
    if config is None:
        config = json.load(open(path, "rb"))
    return config


def get_mongo_client():
    '''
    Used for Sacred to record results
    '''
    config = read_config()
    if "mongo_uri" not in config.keys():
        return pymongo.MongoClient()
    else:
        mongo_uri = config["mongo_uri"]
        return pymongo.MongoClient(mongo_uri)