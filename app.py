from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from api_client import KudaGoAPI
from data_processor import prepare_data
from visualization import create_analytics_graphs
from advanced_analysis import AdvancedAnalysis
from callbacks import register_callbacks


def create_app():
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Культурные мероприятия Екатеринбурга"

    # Загрузка данных с обработкой ошибок
    print("Загрузка данных...")
    try:
        api = KudaGoAPI()
        events = api.get_all_events(max_events=6000)
        if not events.get("results"):
            raise ValueError("Не удалось загрузить данные мероприятий")

        df = prepare_data(events)
        if df.empty:
            raise ValueError("Данные не были корректно обработаны")

    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")

    # Создание макета
    advanced = AdvancedAnalysis(df)
    app.layout = create_layout(df)
    # Регистрация колбэков
    advanced.register_callbacks(app)
    register_callbacks(app, df)

    return app


def create_layout(df):
    """Создание макета приложения"""
    try:
        advanced_tabs = AdvancedAnalysis(df).create_all_tabs()
    except Exception as e:
        print(f"Ошибка при создании расширенного анализа: {e}")
        advanced_tabs = []

    return dbc.Container([
        dcc.Tabs([
            dcc.Tab(label='Аналитика мероприятий', children=[
                dbc.Row([
                    dbc.Col(html.H1("Аналитика культурных мероприятий Екатеринбурга",
                                    className="text-center mb-4"), width=12)
                ]),
                *create_analytics_graphs(df)
            ]),

            dcc.Tab(label='Обзор мероприятий', children=[
                dbc.Row([
                    dbc.Col(html.H1("Культурные мероприятия Екатеринбурга",
                                    className="text-center mb-4"), width=12)
                ]),

                dbc.Row([
                    dbc.Col([
                        html.Label("Фильтр по категориям:"),
                        dcc.Dropdown(
                            id='category-filter',
                            options=[{'label': 'Все категории', 'value': 'all'}] +
                                    [{'label': cat, 'value': cat} for cat in sorted(df['first_category'].unique())],
                            value='all',
                            multi=False,
                            clearable=False,
                            className="mb-3"
                        )
                    ], md=4),

                    dbc.Col([
                        html.Label("Начальная дата (ГГГГ-ММ-ДД):"),
                        dbc.Input(
                            id='start-date-input',
                            type='text',
                            placeholder='Введите дату в формате ГГГГ-ММ-ДД',
                            value=df['date'].min().strftime('%Y-%m-%d') if not df.empty else '',
                            className="mb-2"
                        ),
                        dbc.Alert(id='start-date-alert', color="danger", is_open=False, duration=4000)
                    ], md=4),

                    dbc.Col([
                        html.Label("Конечная дата (ГГГГ-ММ-ДД):"),
                        dbc.Input(
                            id='end-date-input',
                            type='text',
                            placeholder='Введите дату в формате ГГГГ-ММ-ДД',
                            value=df['date'].max().strftime('%Y-%m-%d') if not df.empty else '',
                            className="mb-2"
                        ),
                        dbc.Alert(id='end-date-alert', color="danger", is_open=False, duration=4000)
                    ], md=4)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-scatter",
                            type="default",
                            children=dcc.Graph(
                                id='events-scatter',
                                config={'displayModeBar': False}
                            )
                        )
                    ], width=12)
                ]),

                dbc.Row([
                    dbc.Col([
                        html.Div(id='event-details', className="mt-4 p-3 border rounded",
                                 style={'background-color': '#f8f9fa'})
                    ], width=12)
                ])
            ]),

            *advanced_tabs  # Добавляем все новые вкладки
        ])
    ], fluid=True)


if __name__ == '__main__':
    app = create_app()

    # Универсальный способ запуска
    run_app = getattr(app, 'run', None)
    if callable(run_app):
        run_app(debug=True, port=8050)
    else:
        app.run_server(debug=True, port=8050)