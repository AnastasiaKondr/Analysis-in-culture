import dash_bootstrap_components as dbc
from dash import html


def create_event_card(event_data: dict) -> dbc.Card:
    """Создание подробной карточки события"""
    card_content = [
        create_card_header(event_data),
        create_card_body(event_data)
    ]

    if event_data.get('image'):
        card_content.insert(0, create_card_image(event_data['image']))

    return dbc.Card(
        card_content,
        className="shadow-sm mb-4",
        style={'height': '100%'}
    )


def create_compact_event_card(event_data: dict) -> dbc.Card:
    """Создание компактной карточки события"""
    return dbc.Card([
        dbc.CardHeader(html.Strong(event_data['title'])),
        dbc.CardBody([
            html.P(f"📅 {event_data['date_str']}", className="mb-1"),
            html.P(f"🏛 {event_data['venue']}", className="mb-1"),
            html.P(f"💰 {event_data['price_info']}", className="mb-1"),
            dbc.Button(
                "Подробнее",
                id={'type': 'event-details-btn', 'index': event_data['id']},
                color="primary",
                size="sm",
                className="mt-2"
            )
        ])
    ], className="shadow-sm h-100")


def create_card_image(image_url: str) -> dbc.CardImg:
    """Изображение для карточки"""
    return dbc.CardImg(
        src=image_url,
        top=True,
        style={
            'maxHeight': '300px',
            'objectFit': 'cover',
            'width': '100%'
        }
    )


def create_card_header(event_data: dict) -> dbc.CardHeader:
    """Заголовок карточки"""
    return dbc.CardHeader([
        html.H4(event_data['title'], className="card-title mb-2"),
        html.Div([
            html.Span(
                f"📅 {event_data['date_str']}",
                className="badge bg-light text-dark me-2"
            ),
            html.Span(
                f"🏛 {event_data['venue']}",
                className="badge bg-light text-dark"
            )
        ], className="mb-2"),
        html.Div([
            html.Span(
                f"🏷 {event_data['first_category']}",
                className="badge bg-primary me-2"
            ),
            html.Span(
                f"💰 {event_data['price_info']}",
                className="badge bg-success"
            )
        ])
    ])


def create_card_body(event_data: dict) -> dbc.CardBody:
    """Тело карточки"""
    description = event_data.get('description', 'Описание отсутствует')
    short_description = (description[:200] + '...') if len(description) > 200 else description

    return dbc.CardBody([
        html.H5("Описание мероприятия", className="card-subtitle mb-3"),
        html.P(short_description, className="card-text mb-4"),

        html.Div([
            html.Small(
                f"День недели: {get_weekday_name(event_data['weekday'])}",
                className="text-muted d-block mb-1"
            ),
            html.Small(
                f"Месяц: {get_month_name(event_data['month'])}",
                className="text-muted d-block"
            )
        ])
    ])


def get_weekday_name(weekday_num: int) -> str:
    """Получение названия дня недели"""
    weekdays = ['Понедельник', 'Вторник', 'Среда',
                'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    return weekdays[weekday_num]


def get_month_name(month_num: int) -> str:
    """Получение названия месяца"""
    months = ['Январь', 'Февраль', 'Март', 'Апрель',
              'Май', 'Июнь', 'Июль', 'Август',
              'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
    return months[month_num - 1]