import plotly.express as px
import plotly.graph_objects as go
import calendar
import dash_bootstrap_components as dbc
from dash import dcc
import pandas as pd
from typing import Optional


def create_analytics_graphs(df: pd.DataFrame) -> list:
    """Создание всех аналитических графиков"""
    if df.empty:
        return [dbc.Alert("Нет данных для отображения", color="warning")]

    return [
        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=create_category_distribution(df),
                config={'displayModeBar': False}
            ), md=6),
            dbc.Col(dcc.Graph(
                figure=create_date_distribution(df),
                config={'displayModeBar': False}
            ), md=6)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=create_price_distribution(df),
                config={'displayModeBar': False}
            ), md=6),
            dbc.Col(dcc.Graph(
                figure=create_weekday_distribution(df),
                config={'displayModeBar': False}
            ), md=6)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=create_yearly_distribution(df),
                config={'displayModeBar': False}
            ), width=12)
        ])
    ]


def create_category_distribution(df: pd.DataFrame) -> go.Figure:
    """Распределение мероприятий по категориям"""
    categories_count = df['first_category'].value_counts().reset_index()
    categories_count.columns = ['category', 'count']

    fig = px.bar(
        categories_count,
        x='category',
        y='count',
        title='Распределение мероприятий по категориям',
        labels={'category': 'Категория', 'count': 'Количество'},
        color='category',
        text='count',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        margin=dict(b=100),
        hovermode='x unified'
    )
    return fig


def create_date_distribution(df: pd.DataFrame) -> go.Figure:
    """Распределение мероприятий по датам"""
    fig = px.histogram(
        df,
        x='date',
        title='Распределение мероприятий по датам',
        labels={'date': 'Дата', 'count': 'Количество'},
        nbins=50,
        color_discrete_sequence=['#636EFA']
    )

    fig.update_layout(
        bargap=0.1,
        xaxis_range=[df['date'].min(), df['date'].max()],
        hovermode='x unified'
    )
    return fig


def create_price_distribution(df: pd.DataFrame) -> go.Figure:
    """Распределение мероприятий по ценам"""
    fig = px.box(
        df[df['price'] > 0],
        x='first_category',
        y='price',
        title='Распределение цен по категориям (только платные)',
        labels={'first_category': 'Категория', 'price': 'Цена, руб'},
        color='first_category',
        log_y=True
    )

    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        margin=dict(b=100)
    )
    return fig


def create_weekday_distribution(df: pd.DataFrame) -> go.Figure:
    """Распределение мероприятий по дням недели"""
    weekday_names = list(calendar.day_name)
    weekday_counts = df['weekday'].value_counts().reindex(range(7), fill_value=0)

    fig = px.bar(
        x=weekday_names,
        y=weekday_counts,
        title='Распределение мероприятий по дням недели',
        labels={'x': 'День недели', 'y': 'Количество'},
        color=weekday_names,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(
        showlegend=False,
        hovermode='x unified'
    )
    return fig


def create_yearly_distribution(df: pd.DataFrame) -> go.Figure:
    """Распределение мероприятий по годам"""
    yearly_data = df.groupby('year').size().reset_index(name='count')

    fig = px.line(
        yearly_data,
        x='year',
        y='count',
        title='Динамика мероприятий по годам',
        labels={'year': 'Год', 'count': 'Количество'},
        markers=True,
        text='count'
    )

    fig.update_traces(
        textposition='top center',
        line=dict(width=3),
        marker=dict(size=10)
    )

    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(tickmode='linear')
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: Optional[list] = None) -> go.Figure:
    """Создание тепловой карты корреляций"""
    if columns is None:
        columns = ['price', 'weekday', 'month']

    corr_matrix = df[columns].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        title='Матрица корреляций'
    )

    fig.update_layout(
        xaxis_title='Параметры',
        yaxis_title='Параметры'
    )
    return fig