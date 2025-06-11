import requests
import time
from tqdm import tqdm


class KudaGoAPI:
    BASE_URL = "https://kudago.com/public-api/v1.4/"

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.session = requests.Session()

    def get_all_events(self, city="ekb", max_events=4558, fields=None):
        """Получение всех событий с пагинацией"""

        if fields is None:
            fields = [
                "title", "id", "dates", "categories",
                "description", "price", "is_free",  # Исправленные поля для цены
                "place", "images"
            ]

        all_events = []
        page = 1
        page_size = 100

        with tqdm(total=max_events, desc="Загрузка мероприятий") as pbar:
            while len(all_events) < max_events:
                try:
                    events = self.get_events(
                        city=city,
                        page_size=page_size,
                        page=page,
                        fields=fields
                    )

                    batch = events.get("results", [])
                    if not batch:
                        break

                    remaining = max_events - len(all_events)
                    all_events.extend(batch[:remaining])
                    pbar.update(len(batch[:remaining]))

                    if len(batch) < page_size:
                        break

                    page += 1
                    time.sleep(0.3)

                except requests.exceptions.RequestException as e:
                    print(f"\nОшибка при запросе страницы {page}: {e}")
                    break

        return {"results": all_events}

    def get_events(self, city="ekb", page_size=100, page=1, fields=None):
        """Получение одной страницы событий"""
        url = f"{self.BASE_URL}events/"
        params = {
            "location": city,
            "page_size": page_size,
            "page": page,
            "lang": "ru",
            "order_by": "-dates.start"
        }

        if fields:
            params["fields"] = ",".join(fields)

        if self.api_key:
            params["api_key"] = self.api_key

        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
