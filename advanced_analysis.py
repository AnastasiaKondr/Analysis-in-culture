from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import calendar


def _parse_price(price):
    """Парсинг цены в числовой формат"""
    if isinstance(price, (int, float)):
        price_val = float(price)
        return int(round(price_val))
    if pd.isna(price) or price == 'Цена не указана':
        return 0.0
    if 'бесплатно' in str(price).lower():
        return 0.0


def _create_correlation_tab():
    """Вкладка корреляционного анализа"""
    return dcc.Tab(
        label='Корреляционный анализ',
        children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H4("Настройки анализа"),
                        dcc.Dropdown(
                            id='corr-analysis-type',
                            options=[
                                {'label': 'Цена vs Количество', 'value': 'price-count'},
                                {'label': 'Распределение по дням недели', 'value': 'category-weekday'},
                                {'label': '3D анализ', 'value': '3d-analysis'}
                            ],
                            value='price-count',
                            className='mb-3'
                        )
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(
                        id='corr-main-chart',
                        style={'height': '600px'}  # Увеличиваем высоту графика
                    ), width=12)
                ])
            ], fluid=True, style={'height': '100%'})
        ]
    )


class AdvancedAnalysis:
    def __init__(self, df):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Дополнительная подготовка данных для анализа"""
        # Обработка цены
        self.df['price'] = self.df['price'].apply(_parse_price)
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Временные ряды
        self.time_series = self.df.groupby('date').size()
        self._prepare_regression_data()

    def _prepare_regression_data(self):
        """Подготовка данных для регрессионного анализа"""
        yearly_data = self.df.groupby('year').size().reset_index(name='count')
        self.yearly_data = yearly_data[yearly_data['count'] > 0]

        # Если данных меньше 2 лет, создаем фиктивные
        if len(self.yearly_data) < 2:
            min_year = self.yearly_data['year'].min()
            self.yearly_data = pd.DataFrame({
                'year': [min_year, min_year + 1],
                'count': [self.yearly_data['count'].iloc[0],
                          self.yearly_data['count'].iloc[0] * 1.1]
            })

    def create_all_tabs(self):
        """Создает все вкладки расширенного анализа"""
        tabs = [
            _create_correlation_tab(),
            self._create_regression_tab(),
            self._create_kpi_tab()
        ]

        # Добавляем вкладку прогноза только если достаточно данных
        forecast_tab = self._create_forecast_tab()
        if forecast_tab is not None:
            tabs.append(forecast_tab)

        return tabs

    def _create_regression_tab(self):
        """Вкладка регрессионного анализа"""
        if len(self.yearly_data) < 2:
            return dcc.Tab(
                label='Регрессионный анализ',
                children=[
                    dbc.Alert("Недостаточно данных для анализа (требуется минимум 2 года)",
                              color="warning")
                ]
            )

        X = self.yearly_data['year'].values.reshape(-1, 1)
        y = self.yearly_data['count'].values

        model = LinearRegression()
        model.fit(X, y)

        fig = px.scatter(
            self.yearly_data,
            x='year',
            y='count',
            trendline="ols",
            title="Тренд количества мероприятий",
            labels={'year': 'Год', 'count': 'Количество'}
        )

        return dcc.Tab(
            label='Регрессионный анализ',
            children=[
                dbc.Container([
                    dbc.Row(dcc.Graph(figure=fig)),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Статистика модели:"),
                            html.P(f"Коэффициент (наклон): {model.coef_[0]:.2f}"),
                            html.P(f"Пересечение: {model.intercept_:.2f}"),
                            html.P(f"Тренд: {'возрастающий' if model.coef_[0] > 0 else 'убывающий'}")
                        ])
                    ])
                ])
            ]
        )

    def _create_kpi_tab(self):
        """Вкладка ключевых показателей"""
        # Расчет KPI
        self.df['price'] = self.df['price'].apply(_parse_price)
        total_events = len(self.df)
        valid_prices = self.df[(self.df['price'] > 0)]['price']
        avg_price = valid_prices.mean()

        if avg_price >= 1000000:
            avg_price_str = f"{avg_price / 1000000:.1f}М руб"  # 1.2М руб
        elif avg_price >= 1000:
            avg_price_str = f"{avg_price / 1000:.1f}K руб"  # 1.2K руб
        else:
            avg_price_str = f"{avg_price:.0f} руб"  # 1200 руб

        free_events = len(self.df[self.df['price'] == 0])

        weekday_names = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
        weekday_counts = self.df['weekday'].value_counts().sort_index()
        weekday_fig = px.bar(
            x=weekday_names,
            y=weekday_counts,
            title='Распределение по дням недели',
            labels={'x': 'День недели', 'y': 'Количество'},
            color=weekday_names
        )
        weekday_fig.update_layout(showlegend=False)

        category_dropdown = dcc.Dropdown(
            id='profitability-category-filter',
            options=[{'label': 'Все категории', 'value': 'all'}] +
                    [{'label': cat, 'value': cat} for cat in sorted(self.df['first_category'].unique())],
            value='all',
            clearable=False,
            className='mb-3'
        )

        return dcc.Tab(
            label='Ключевые показатели',
            children=[
                dbc.Container([
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Всего мероприятий"),
                            dbc.CardBody(f"{total_events}")
                        ]), width=4),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Средняя цена"),
                            dbc.CardBody(avg_price_str)
                        ]), width=4),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Бесплатных мероприятий"),
                            dbc.CardBody(f"{free_events} ({free_events / total_events:.1%})")
                        ]), width=4)
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(
                            figure=px.pie(
                                self.df,
                                names='first_category',
                                title='Распределение по категориям'
                            )
                        ), width=6),
                        dbc.Col(dcc.Graph(
                            figure=weekday_fig  # Используем обновленный график
                        ), width=6),
                        dbc.Row([
                            dbc.Col([
                                html.H4("Анализ доходности по категориям"),
                                category_dropdown
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(
                                id='profitability-chart',
                                figure=self._create_profitability_chart()
                            ), width=12)
                        ])
                    ])
                ])
            ]
        )

    def _create_forecast_tab(self):
        """Вкладка прогнозирования"""
        if len(self.time_series) < 24:  # Минимум 2 года данных
            return None

        try:
            # Создаем базовый график с последними 6 годами данных
            last_6_years = self.time_series[
                self.time_series.index >= (self.time_series.index.max() - pd.DateOffset(years=6))]

            base_fig = px.line(
                x=last_6_years.index,
                y=last_6_years.values,
                title='Прогноз количества мероприятий (последние 6 лет)',
                labels={'x': 'Дата', 'y': 'Количество'}
            )

            # Создаем график для исторического прогноза (2016-2020)
            historical_data = self.time_series[
                (self.time_series.index >= pd.to_datetime('2016-01-01')) &
                (self.time_series.index < pd.to_datetime('2021-01-01'))]

            if len(historical_data) >= 24:  # Если есть достаточно данных
                hist_fig = px.line(
                    x=historical_data.index,
                    y=historical_data.values,
                    title='Исторический прогноз (2016-2020 данные)',
                    labels={'x': 'Дата', 'y': 'Количество'}
                )
            else:
                hist_fig = px.line(title='Недостаточно данных для исторического прогноза')

            # Добавляем кнопки для выбора периода прогноза
            forecast_controls = dbc.Row([
                dbc.Col([
                    html.Label("Период прогноза (лет):"),
                    dcc.Slider(
                        id='forecast-period-slider',
                        min=1,
                        max=5,
                        step=1,
                        value=2,
                        marks={i: str(i) for i in range(1, 6)}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Метод прогнозирования:"),
                    dcc.Dropdown(
                        id='forecast-method',
                        options=[
                            {'label': 'Линейная регрессия', 'value': 'linear'},
                            {'label': 'Средний рост', 'value': 'average'},
                            {'label': 'Сезонность + тренд', 'value': 'seasonal'}
                        ],
                        value='linear',
                        clearable=False
                    )
                ], width=6)
            ])

            return dcc.Tab(
                label='Прогнозирование',
                children=[
                    dbc.Container([
                        dbc.Row(forecast_controls),
                        dbc.Row(dcc.Graph(id='forecast-chart', figure=base_fig)),
                        dbc.Row([
                            dbc.Col([
                                html.Div(id='forecast-stats')
                            ])
                        ]),
                        dbc.Row(dcc.Graph(id='historical-forecast-chart', figure=hist_fig)),
                        dbc.Row([
                            dbc.Col([
                                html.Div(id='historical-forecast-stats')
                            ])
                        ])
                    ], fluid=True)
                ]
            )
        except Exception as e:
            print(f"Ошибка при создании вкладки прогнозирования: {e}")
            return None

    def _create_price_count_chart(self):
        monthly_data = self.df.groupby(['year', 'month']).agg({
            'price': 'mean',
            'id': 'count'
        }).reset_index()
        return px.scatter(
            monthly_data,
            x='id',
            y='price',
            trendline="ols",
            title="Зависимость цены от количества мероприятий",
            labels={'id': 'Количество', 'price': 'Цена (руб)'}
        )

    def _create_category_weekday_chart(self):
        weekday_names = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
        weekday_data = self.df.groupby(['first_category', 'weekday']).size().unstack()
        weekday_data.columns = weekday_names  # Переименовываем столбцы

        fig = px.imshow(
            weekday_data,
            labels=dict(x="День недели", y="Категория", color="Количество"),
            title="Распределение мероприятий по категориям и дням недели",
            height=600  # Увеличиваем высоту графика
        )
        return fig

    def _create_3d_analysis_chart(self):
        monthly_data = self.df.groupby(['year', 'month']).agg({
            'price': 'mean',
            'id': 'count'
        }).reset_index()
        return px.scatter_3d(
            monthly_data,
            x='year',
            y='month',
            z='id',
            color='price',
            title="3D анализ: год/месяц/количество (цвет=цена)",
            labels={'year': 'Год', 'month': 'Месяц', 'id': 'Количество'}
        )

    def _create_profitability_chart(self, selected_category: str = None) -> go.Figure:
        """Создает график доходности по месяцам для выбранной категории"""
        from plotly.subplots import make_subplots

        df = self.df.copy()
        if selected_category and selected_category != 'all':
            df = df[df['first_category'] == selected_category]

        # Группируем данные
        monthly_data = df.groupby(['year', 'month']).agg({
            'price': ['mean', 'count', lambda x: (x > 0).sum()]
        }).reset_index()
        monthly_data.columns = ['year', 'month', 'avg_price', 'total_events', 'paid_events']

        # Рассчитываем доход
        paid_df = df[df['price'] > 0]
        if not paid_df.empty:
            monthly_income = paid_df.groupby(['year', 'month'])['price'].sum().reset_index()
            monthly_data = monthly_data.merge(monthly_income, on=['year', 'month'], how='left')
            monthly_data['total_income'] = monthly_data['price'].fillna(0)
        else:
            monthly_data['total_income'] = 0

        monthly_data['month_name'] = monthly_data['month'].apply(lambda x: calendar.month_abbr[x])
        unique_years = sorted(monthly_data['year'].unique())

        # Ограничиваем количество отображаемых лет (макс. 5)
        if len(unique_years) > 5:
            unique_years = unique_years[-5:]  # Берем последние 5 лет
            monthly_data = monthly_data[monthly_data['year'].isin(unique_years)]

        # Настройки для разного количества лет
        if len(unique_years) > 1:
            rows = len(unique_years)
            vertical_spacing = min(0.3, 0.9 / rows)  # Автоподбор spacing

            fig = make_subplots(
                rows=rows, cols=1,
                subplot_titles=[f"Год {year}" for year in unique_years],
                shared_xaxes=True,
                vertical_spacing=vertical_spacing
            )

            for i, year in enumerate(unique_years, 1):
                year_data = monthly_data[monthly_data['year'] == year]

                fig.add_trace(
                    go.Bar(
                        x=year_data['month_name'],
                        y=year_data['avg_price'],
                        name='Средняя цена',
                        marker_color='#636EFA',
                        legendgroup='price',
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=year_data['month_name'],
                        y=year_data['total_income'],
                        name='Общий доход',
                        line=dict(color='#00CC96', width=3),
                        yaxis=f'y{(i * 2)}',
                        legendgroup='income',
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )

            # Общие настройки
            fig.update_layout(
                title_text=f'Доходность мероприятий {f"({selected_category})" if selected_category else ""}',
                hovermode='x unified',
                height=200 * rows,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )

            # Настройка осей для каждого subplot
            for i in range(1, rows + 1):
                fig.update_yaxes(
                    title_text="Средняя цена (руб)",
                    row=i, col=1
                )
                fig.update_yaxes(
                    title_text="Общий доход (руб)",
                    overlaying=f'y{i * 2 - 1}',
                    side='right',
                    row=i, col=1
                )

        else:
            # График для одного года
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=monthly_data['month_name'],
                y=monthly_data['avg_price'],
                name='Средняя цена',
                marker_color='#636EFA'
            ))

            fig.add_trace(go.Scatter(
                x=monthly_data['month_name'],
                y=monthly_data['total_income'],
                name='Общий доход',
                line=dict(color='#00CC96', width=3),
                yaxis='y2'
            ))

            fig.update_layout(
                title_text=f'Доходность мероприятий {f"({selected_category})" if selected_category else ""}',
                xaxis_title='Месяц',
                yaxis=dict(title='Средняя цена (руб)'),
                yaxis2=dict(
                    title='Общий доход (руб)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified'
            )

        return fig

    def register_callbacks(self, app):
        @app.callback(
            Output('corr-main-chart', 'figure'),
            [Input('corr-analysis-type', 'value')]
        )
        def update_correlation_chart(analysis_type):
            if analysis_type == 'price-count':
                return self._create_price_count_chart()
            elif analysis_type == 'category-weekday':
                return self._create_category_weekday_chart()
            else:
                return self._create_3d_analysis_chart()

        @app.callback(
            [Output('historical-forecast-chart', 'figure'),
             Output('historical-forecast-stats', 'children')],
            [Input('forecast-period-slider', 'value'),
             Input('forecast-method', 'value')]
        )
        def update_historical_forecast(period_years, method):
            # Инициализируем переменные по умолчанию
            fig = go.Figure()
            stats = [html.P("Неизвестный метод прогнозирования")]

            try:
                # Берем данные за 2016-2020 годы
                historical_data = self.time_series[
                    (self.time_series.index >= pd.to_datetime('2016-01-01')) &
                    (self.time_series.index < pd.to_datetime('2021-01-01'))
                    ]

                if len(historical_data) < 24:
                    return (
                        px.line(title="Недостаточно данных для исторического прогноза"),
                        dbc.Alert("Недостаточно данных для исторического прогноза", color="warning")
                    )

                # Разделяем на тренировочные (2016-2018) и тестовые (2019-2020) данные
                train_data = historical_data[historical_data.index < pd.to_datetime('2019-01-01')]
                test_data = historical_data[historical_data.index >= pd.to_datetime('2019-01-01')]

                # Создаем базовый график
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=train_data.index,
                    y=train_data.values,
                    name='Тренировочные данные (2016-2018)',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=test_data.index,
                    y=test_data.values,
                    name='Реальные данные (2019-2020)',
                    line=dict(color='green')
                ))

                if method == 'average':
                    # Метод среднего роста
                    monthly_growth = train_data.pct_change().mean()
                    last_value = train_data.iloc[-1]

                    forecast_dates = pd.date_range(
                        start=pd.to_datetime('2019-01-01'),
                        end=pd.to_datetime('2020-12-31'),
                        freq='ME'
                    )

                    forecast_values = [last_value * (1 + monthly_growth) ** i for i in
                                       range(1, len(forecast_dates) + 1)]

                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        name='Прогноз (средний рост)',
                        line=dict(color='orange', dash='dash')
                    ))

                    # Вычисляем ошибку прогноза
                    actual_values = test_data.reindex(forecast_dates).fillna(0)
                    mae = np.mean(np.abs(forecast_values - actual_values))

                    stats = [
                        html.H5("Метод среднего роста:"),
                        html.P(f"Среднемесячный рост: {monthly_growth:.1%}"),
                        html.P(f"Средняя ошибка прогноза: {mae:.1f} мероприятий в месяц"),
                        html.P(f"Прогноз на {len(forecast_dates)} месяцев")
                    ]

                elif method == 'linear':
                    # Линейная регрессия
                    X = np.arange(len(train_data)).reshape(-1, 1)
                    y = train_data.values
                    model = LinearRegression()
                    model.fit(X, y)

                    forecast_dates = pd.date_range(
                        start=pd.to_datetime('2019-01-01'),
                        end=pd.to_datetime('2020-12-31'),
                        freq='ME'
                    )
                    forecast_X = np.arange(len(train_data), len(train_data) + len(forecast_dates)).reshape(-1, 1)
                    forecast_values = model.predict(forecast_X)

                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        name='Прогноз (линейный)',
                        line=dict(color='red', dash='dash')
                    ))

                    actual_values = test_data.reindex(forecast_dates).fillna(0)
                    mae = np.mean(np.abs(forecast_values - actual_values))

                    stats = [
                        html.H5("Линейная регрессия:"),
                        html.P(f"Наклон: {model.coef_[0]:.2f} мероприятий/месяц"),
                        html.P(f"Средняя ошибка прогноза: {mae:.1f} мероприятий в месяц"),
                        html.P(f"Прогноз на {len(forecast_dates)} месяцев")
                    ]

                else:  # seasonal
                    # Сезонная декомпозиция
                    try:
                        decomposition = seasonal_decompose(train_data, model='additive', period=12)

                        trend = decomposition.trend.dropna()
                        X = np.arange(len(trend)).reshape(-1, 1)
                        y = trend.values

                        trend_model = LinearRegression()
                        trend_model.fit(X, y)

                        forecast_dates = pd.date_range(
                            start=pd.to_datetime('2019-01-01'),
                            end=pd.to_datetime('2020-12-31'),
                            freq='ME'
                        )
                        forecast_X = np.arange(len(trend), len(trend) + len(forecast_dates)).reshape(-1, 1)
                        future_trend = trend_model.predict(forecast_X)

                        seasonal = decomposition.seasonal[-12:].values
                        future_seasonal = np.tile(seasonal, int(np.ceil(len(forecast_dates) / 12)))[
                                          :len(forecast_dates)]

                        forecast_values = future_trend.flatten() + future_seasonal

                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast_values,
                            name='Прогноз (сезонность+тренд)',
                            line=dict(color='purple', dash='dash')
                        ))

                        actual_values = test_data.reindex(forecast_dates).fillna(0)
                        mae = np.mean(np.abs(forecast_values - actual_values))

                        stats = [
                            html.H5("Сезонность + тренд:"),
                            html.P(
                                f"Тренд: {'↑' if trend_model.coef_[0] > 0 else '↓'} {abs(trend_model.coef_[0]):.2f}/мес"),
                            html.P(f"Средняя ошибка прогноза: {mae:.1f} мероприятий в месяц"),
                            html.P(f"Прогноз на {len(forecast_dates)} месяцев")
                        ]
                    except Exception as e:
                        stats = [
                            dbc.Alert(f"Ошибка сезонной декомпозиции: {str(e)}", color="danger")
                        ]

                fig.update_layout(
                    title=f'Исторический прогноз (2016-2020) - метод {method}',
                    xaxis_title='Дата',
                    yaxis_title='Количество мероприятий',
                    hovermode='x unified'
                )

            except Exception as e:
                fig = px.line(title=f"Ошибка: {str(e)}")
                stats = [
                    dbc.Alert(f"Ошибка при построении прогноза: {str(e)}", color="danger")
                ]

            return fig, stats

        @app.callback(
            Output('profitability-chart', 'figure'),
            [Input('profitability-category-filter', 'value')]
        )
        def update_profitability_chart(selected_category):
            return self._create_profitability_chart(selected_category)
