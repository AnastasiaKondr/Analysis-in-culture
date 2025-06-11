from dash import html

from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from visualization import create_event_card
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose


def register_callbacks(app, df):
    @app.callback(
        [Output('events-scatter', 'figure'),
         Output('start-date-alert', 'is_open'),
         Output('start-date-alert', 'children'),
         Output('end-date-alert', 'is_open'),
         Output('end-date-alert', 'children')],
        [Input('category-filter', 'value'),
         Input('start-date-input', 'value'),
         Input('end-date-input', 'value')]
    )
    def update_scatter(selected_category, start_date, end_date):
        start_date_error = None
        end_date_error = None

        try:
            start_date = pd.to_datetime(start_date)
        except ValueError:
            start_date_error = "Неверный формат даты. Используйте ГГГГ-ММ-ДД"

        try:
            end_date = pd.to_datetime(end_date)
        except ValueError:
            end_date_error = "Неверный формат даты. Используйте ГГГГ-ММ-ДД"

        if start_date_error or end_date_error:
            return (
                px.scatter(title="Ошибка в параметрах фильтрации"),
                bool(start_date_error),
                start_date_error or "",
                bool(end_date_error),
                end_date_error or ""
            )

        # Фильтрация данных
        filtered_df = df.copy()
        filtered_df = filtered_df[
            (filtered_df['date'] >= start_date) &
            (filtered_df['date'] <= end_date)
            ]

        if selected_category != 'all':
            filtered_df = filtered_df[filtered_df['first_category'] == selected_category]

        # Создание графика
        fig = px.scatter(
            filtered_df,
            x='date',
            y='first_category',
            color='first_category',
            hover_data=['title', 'venue', 'price'],
            title=f"Мероприятия с {start_date.date()} по {end_date.date()}",
            labels={
                'date': 'Дата',
                'first_category': 'Категория'
            }
        )

        fig.update_traces(
            marker=dict(size=12, opacity=0.7),
            selector=dict(mode='markers')
        )

        return fig, False, "", False, ""

    @app.callback(
        Output('event-details', 'children'),
        [Input('events-scatter', 'clickData')],
        [State('category-filter', 'value'),
         State('start-date-input', 'value'),
         State('end-date-input', 'value')]
    )
    def display_event_details(clickData, selected_category, start_date, end_date):
        if not clickData:
            return dbc.Alert("Выберите мероприятие на графике", color="info")

        point = clickData['points'][0]
        event_title = point['customdata'][0]

        # Находим событие с учетом текущих фильтров
        filtered_df = df.copy()
        try:
            filtered_df = filtered_df[
                (filtered_df['date'] >= pd.to_datetime(start_date)) &
                (filtered_df['date'] <= pd.to_datetime(end_date))
                ]
        except:
            pass

        if selected_category != 'all':
            filtered_df = filtered_df[filtered_df['first_category'] == selected_category]

        event_data = filtered_df[filtered_df['title'] == event_title]

        if event_data.empty:
            return dbc.Alert("Данные о мероприятии не найдены", color="warning")

        return create_event_card(event_data.iloc[0])

    @app.callback(
        [Output('forecast-chart', 'figure'),
         Output('forecast-stats', 'children')],
        [Input('forecast-period-slider', 'value'),
         Input('forecast-method', 'value')],
        [State('category-filter', 'value'),
         State('start-date-input', 'value'),
         State('end-date-input', 'value')]
    )
    def update_forecast(period_years, method, selected_category, start_date, end_date):
        # Фильтрация данных по выбранным датам и категории
        filtered_df = df.copy()
        try:
            filtered_df = filtered_df[
                (filtered_df['date'] >= pd.to_datetime(start_date)) &
                (filtered_df['date'] <= pd.to_datetime(end_date))
                ]
        except:
            pass

        if selected_category != 'all':
            filtered_df = filtered_df[filtered_df['first_category'] == selected_category]

        # Подготовка временного ряда
        time_series = filtered_df.groupby('date').size()

        if len(time_series) < 24:  # Минимум 2 года данных
            return (
                px.line(title="Недостаточно данных для прогноза (требуется минимум 2 года)"),
                dbc.Alert("Недостаточно данных для прогноза (требуется минимум 2 года)", color="warning")
            )

        # Берем последние 6 лет данных
        last_date = time_series.index.max()
        start_date_6y = last_date - pd.DateOffset(years=6)
        historical_data = time_series[time_series.index >= start_date_6y]

        # Создаем базовый график
        fig = px.line(
            x=historical_data.index,
            y=historical_data.values,
            title=f'Прогноз количества мероприятий на {period_years} лет (последние 6 лет)',
            labels={'x': 'Дата', 'y': 'Количество'}
        )

        # Генерируем прогноз в зависимости от выбранного метода
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=period_years * 12,
            freq='ME'
        )

        if method == 'linear':
            # Линейная регрессия по годам
            yearly_data = time_series.resample('YE').count()
            X = np.arange(len(yearly_data)).reshape(-1, 1)
            y = yearly_data.values

            model = LinearRegression()
            model.fit(X, y)

            # Прогнозируем будущие значения
            future_X = np.arange(len(yearly_data), len(yearly_data) + period_years).reshape(-1, 1)
            future_y = model.predict(future_X)

            # Добавляем прогноз на график
            forecast_values = np.linspace(future_y[0], future_y[-1], len(forecast_dates))
            fig.add_scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines',
                name='Прогноз (линейный)',
                line=dict(color='red', dash='dash')
            )

            stats = [
                html.H5("Статистика линейного прогноза:"),
                html.P(f"Коэффициент роста: {model.coef_[0]:.2f} мероприятий/год"),
                html.P(f"Прогноз на {period_years} лет вперед: {future_y[-1]:.0f} мероприятий/год")
            ]

        elif method == 'average':
            # Средний рост за последние годы
            yearly_data = time_series.resample('YE').count()
            growth_rates = yearly_data.pct_change().dropna()
            avg_growth = growth_rates.mean()

            last_value = yearly_data.iloc[-1]
            forecast_values = [last_value * (1 + avg_growth) ** i for i in range(1, period_years + 1)]

            # Добавляем прогноз на график
            forecast_yearly_dates = pd.date_range(
                start=last_date + pd.DateOffset(years=1),
                periods=period_years,
                freq='YE'
            )

            fig.add_scatter(
                x=forecast_yearly_dates,
                y=forecast_values,
                mode='lines+markers',
                name='Прогноз (средний рост)',
                line=dict(color='green', dash='dash')
            )

            stats = [
                html.H5("Статистика прогноза по среднему росту:"),
                html.P(f"Средний годовой рост: {avg_growth:.1%}"),
                html.P(f"Прогноз на {period_years} лет вперед: {forecast_values[-1]:.0f} мероприятий/год")
            ]

        else:  # seasonal
            # Декомпозиция временного ряда
            resampled = time_series.resample('ME').mean().ffill()
            try:
                decomposition = seasonal_decompose(resampled, model='additive', period=12)

                # Прогнозируем тренд
                trend = decomposition.trend.dropna()
                X = np.arange(len(trend)).reshape(-1, 1)
                y = trend.values

                trend_model = LinearRegression()
                trend_model.fit(X, y)

                # Прогнозируем будущий тренд
                future_X = np.arange(len(trend), len(trend) + period_years * 12).reshape(-1, 1)
                future_trend = trend_model.predict(future_X)

                # Добавляем сезонность
                seasonal = decomposition.seasonal[-12:].values  # Берем последний год сезонности
                future_seasonal = np.tile(seasonal, period_years)

                # Комбинируем тренд и сезонность
                forecast_values = future_trend + future_seasonal

                # Добавляем прогноз на график
                fig.add_scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    mode='lines',
                    name='Прогноз (сезонность+тренд)',
                    line=dict(color='purple', dash='dash')
                )

                stats = [
                    html.H5("Статистика прогноза с сезонностью:"),
                    html.P(f"Тренд: {'возрастающий' if trend_model.coef_[0] > 0 else 'убывающий'}"),
                    html.P(f"Средняя сезонная амплитуда: {np.abs(seasonal).mean():.1f} мероприятий"),
                    html.P(f"Прогноз на {period_years} лет вперед: {forecast_values[-1]:.0f} мероприятий")
                ]
            except Exception as e:
                return (
                    px.line(title=f"Ошибка при декомпозиции временного ряда: {str(e)}"),
                    dbc.Alert(f"Ошибка при декомпозиции временного ряда: {str(e)}", color="danger")
                )

        return fig, stats