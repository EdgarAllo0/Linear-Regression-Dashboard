# Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import random

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@st.cache_resource
def TwoLinesPlot(
        series1: pd.Series | None,
        series2: pd.Series | None,
        title: str,
        axis1: str,
        axis2: str,
        shared_y: bool = True,
):
    # Colors
    palette = list(sns.color_palette("Spectral", 20).as_hex())

    # Conditions for plotting
    condition_1 = series1 is None or series1.empty or series1.isnull().all()
    condition_2 = series2 is None or series2.empty or series2.isnull().all()

    if not condition_1 and not condition_2:

        if shared_y:
            fig = make_subplots(
                specs=[
                    [
                        {"secondary_y": True}
                    ]
                ],
            )

            fig.add_trace(
                go.Scatter(
                    x=series1.index,
                    y=series1,
                    name=axis1,
                    marker=dict(
                        color=random.choice(palette)
                    )
                ),
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=series2.index,
                    y=series2,
                    name=axis2,
                    marker=dict(
                        color=random.choice(palette)
                    )
                ),
                secondary_y=True
            )

            fig.update_yaxes(
                title_text=axis1,
                title_font=dict(size=18),
                secondary_y=False
            )

            fig.update_yaxes(
                title_text=axis2,
                title_font=dict(size=18),
                secondary_y=True
            )

            fig.update_xaxes(
                title_text='Date',
                title_font=dict(size=18)
            )

            fig.update_xaxes(
                rangeslider_visible=False
            )

            fig.update_layout(
                title={
                    'text': title,
                    'font': {
                        'size': 25
                    }
                },
                title_x=0.5,
                title_xanchor="center"
            )

            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    xanchor="center",
                    y=1.0,
                    x=0.5,
                    font=dict(
                        size=20,
                        color="black"
                    )
                )
            )

            fig.update_layout(
                font=dict(
                    size=20,
                ),
                legend=dict(
                    font=dict(
                        size=20,
                    )
                )
            )

            fig.update_layout(
                height=600
            )

            fig.update_layout(
                font={
                    'size': 20
                }
            )

        else:

            fig = make_subplots()

            fig.add_trace(
                go.Scatter(
                    x=series1.index,
                    y=series1,
                    name=axis1,
                    marker=dict(
                        color=random.choice(palette)
                    )
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=series2.index,
                    y=series2,
                    name=axis2,
                    marker=dict(
                        color=random.choice(palette)
                    )
                ),
            )

            fig.update_yaxes(
                title_text=axis1,
                title_font=dict(size=18),
            )

            fig.update_xaxes(
                title_text='Date',
                title_font=dict(size=18)
            )

            fig.update_xaxes(
                rangeslider_visible=False
            )

            fig.update_layout(
                title={
                    'text': title,
                    'font': {
                        'size': 25
                    }
                },
                title_x=0.5,
                title_xanchor="center"
            )

            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    xanchor="center",
                    y=1.0,
                    x=0.5,
                    font=dict(
                        size=20,
                        color="black"
                    )
                )
            )

            fig.update_layout(
                font=dict(
                    size=20,
                ),
                legend=dict(
                    font=dict(
                        size=20,
                    )
                )
            )

            fig.update_layout(
                height=600
            )

            fig.update_layout(
                font={
                    'size': 20
                }
            )

    else:

        fig = None
        st.warning('There was an error with Series Data...', icon="⚠️")

    return fig
