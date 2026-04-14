import json
import os
import shelve
from typing import Any


class CacheFileManager:
    def __init__(self, cache_path: str, from_jsonl: str=None, from_json: str=None):
        self.cache_path = cache_path
        self.cache_dir = os.path.dirname(cache_path)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self._cache = shelve.open(cache_path, writeback=True)

        if from_jsonl is not None:
            self._from_jsonl(from_jsonl)

        if from_json is not None:
            self._from_json(from_json)

    @property
    def cache(self):
        return self._cache

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value
    
    def __getitem__(self, key: str) -> Any:
        return self._cache[key]

    def sync(self):
        self._cache.sync()

    def _from_jsonl(self, jsonl_path: str):
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self._cache[data["topic"]] = data

    def _from_json(self, json_path: str):
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
            for topic, data in data_dict.items():
                self._cache[topic] = data

    def to_json(self):
        json_path = self.cache_path + ".json"
        with open(json_path, 'w') as f:
            json.dump(dict(self._cache), f, indent=2, ensure_ascii=False)

    def to_json4corr(self):
        json_path = self.cache_path + "_corrected.json"
        with open(json_path, 'w') as f:
            json.dump(dict(self._cache), f, indent=2, ensure_ascii=False)

    def __del__(self):
        self._cache.close()