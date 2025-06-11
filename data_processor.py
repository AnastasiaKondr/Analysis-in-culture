import pandas as pd
from datetime import datetime


def is_valid_timestamp(timestamp):
    """Проверка валидности timestamp"""
    try:
        ts = int(timestamp)
        return 946684800 <= ts <= 1893456000  # 2000-2030
    except (ValueError, TypeError):
        return False


def format_timestamp(timestamp):
    """Форматирование timestamp в дату"""
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d")


def prepare_data(events):
    """Основная функция подготовки данных"""
    if not events.get("results"):
        return pd.DataFrame()

    processed_events = []

    for event in events["results"]:
        try:
            if validate_event_data(event):
                event_data = process_single_event(event)
                # Добавляем числовое представление цены
                event_data['price'] = parse_price(event)
                processed_events.append(event_data)
        except Exception as e:
            print(f"Ошибка обработки события: {e}")
            continue

    return pd.DataFrame(processed_events)


def parse_price(event):
    """Парсинг цены в числовой формат"""
    if isinstance(event["price"], str):
        price_str = ''.join(filter(str.isdigit, event["price"]))
        if price_str:
            return int(price_str)
        else:
            return 0.0  # Пропускаем, если не удалось извлечь цену
    elif isinstance(event["price"], (int, float)):
        return event["price"]
    else:
        return 0.0


def validate_event_data(event):
    """Валидация данных события"""
    return (event.get("dates") and
            isinstance(event["dates"], list) and
            len(event["dates"]) > 0 and
            event["dates"][0].get("start") and
            is_valid_timestamp(event["dates"][0]["start"]))


def process_single_event(event):
    """Обработка одного события"""
    timestamp = event["dates"][0]["start"]
    date_obj = datetime.fromtimestamp(int(timestamp))

    # Определяем цену
    price = parse_price(event)
    price_info = "Бесплатно" if price == 0 else f"{price} руб" if price > 0 else "Цена не указана"

    return {
        "id": event.get("id"),
        "title": event.get("title", "Без названия"),
        "date": date_obj,
        "date_str": format_timestamp(timestamp),
        "categories": format_categories(event.get("categories")),
        "first_category": get_first_category(event.get("categories")),
        "description": format_description(event.get("description")),
        "price": price,
        "price_info": price_info,
        "venue": get_venue(event.get("place")),
        "image": get_image(event.get("images")),
        "weekday": date_obj.weekday(),
        "month": date_obj.month,
        "year": date_obj.year
    }


def format_categories(categories):
    """Форматирование категорий"""
    return ", ".join(categories) if categories else "Без категории"


def get_first_category(categories):
    """Получение первой категории"""
    return categories[0] if categories else "Без категории"


def format_description(description):
    """Форматирование описания"""
    if not description:
        return "Описание отсутствует"
    return (description[:500] + "...") if len(description) > 500 else description


def get_venue(place):
    """Получение места проведения"""
    return place.get("title", "Место не указано") if place else "Место не указано"


def get_image(images):
    """Получение изображения"""
    return images[0].get("image") if images and len(images) > 0 else None