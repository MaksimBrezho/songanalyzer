import requests
import random
import time

class ProxyManager:
    PROXY_URL = "https://www.proxy-list.download/api/v1/get?type=https&anon=elite"

    def __init__(self):
        self.proxies = []
        self.last_fetch_time = 0
        self.fetch_interval = 300  # Обновляем список каждые 5 минут

    def _fetch_proxies(self):
        try:
            response = requests.get(self.PROXY_URL, timeout=10)
            if response.ok:
                self.proxies = response.text.strip().split('\n')
                random.shuffle(self.proxies)
                self.last_fetch_time = time.time()
        except Exception:
            pass  # В случае ошибки просто не обновляем

    def get_proxy(self):
        if not self.proxies or time.time() - self.last_fetch_time > self.fetch_interval:
            self._fetch_proxies()
        return self.proxies.pop() if self.proxies else None
