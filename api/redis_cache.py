import redis
import os
import json
import hashlib
from src.logger import logging

class RedisCache:
    def __init__(self):
        # We will set this environment variable later when we deploy
        self.redis_url = os.getenv("UPSTASH_REDIS_URL", None)
        self.redis_client = None
        
        if self.redis_url:
            try:
                # decode_responses=True ensures we get strings, not byte strings
                self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
                self.redis_client.ping()
                logging.info("Successfully connected to Upstash Redis!")
            except Exception as e:
                logging.warning(f"Redis connection failed, bypassing cache. Error: {e}")
                self.redis_client = None
        else:
            logging.warning("UPSTASH_REDIS_URL not found in .env. Cache disabled.")

    def generate_cache_key(self, input_dict: dict) -> str:
        # Sort keys to ensure {"a":1, "b":2} and {"b":2, "a":1} generate the same hash
        dict_string = json.dumps(input_dict, sort_keys=True)
        return "car_price_" + hashlib.sha256(dict_string.encode()).hexdigest()

    def get_cached_prediction(self, cache_key: str):
        if not self.redis_client:
            return None
        try:
            cached_price = self.redis_client.get(cache_key)
            if cached_price:
                logging.info(f"Cache Hit! Retrieved price for {cache_key}")
                return float(cached_price)
        except Exception as e:
            logging.error(f"Redis GET error: {e}")
        return None

    def set_cached_prediction(self, cache_key: str, price: float, ttl_seconds: int = 86400):
        # Default TTL is 24 hours (86400 seconds)
        if not self.redis_client:
            return
        try:
            self.redis_client.setex(cache_key, ttl_seconds, price)
            logging.info(f"Saved prediction to cache: {cache_key}")
        except Exception as e:
            logging.error(f"Redis SET error: {e}")