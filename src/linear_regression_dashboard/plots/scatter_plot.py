# Libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm


@st.cache_resource
def AdjustableScatterPlot(
        y_series: pd.Series,
        x_series: pd.Series,
        b0: int,
        b1: int,
):

    # Get the supposed trend line
    trend_y_input = b0 + b1 * x_series

    # Get the actual trend line
    model = sm.OLS(
        y_series,
        sm.add_constant(x_series)
    )

    results = model.fit()

    b0_ols, b1_ols = results.params

    trend_y_ols = b0_ols + b1_ols * x_series

    # Make the Scatter Plot
    scatter = go.Scatter(
        x=x_series,
        y=y_series,
        mode='markers',
        name='data'
    )

    trend_line_input = go.Scatter(
        x=x_series,
        y=trend_y_input,
        mode='lines',
        name='Inferred Trend Line'
    )

    trend_line_ols = go.Scatter(
        x=x_series,
        y=trend_y_ols,
        mode='lines',
        name='Real Trend Line'
    )

    # Set the figure
    fig = go.Figure()
    fig.add_trace(scatter)
    fig.add_trace(trend_line_input)
    fig.add_trace(trend_line_ols)

    annotation_text = f"OLS Betas:<br>b0 = {b0_ols:.2f}<br>b1 = {b1_ols:.2f}"
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.95, y=0.95,
        text=annotation_text,
        showarrow=False,
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        title='Scatter Plot with Trend Lines',
        xaxis_title='Economic Growth Rate',
        yaxis_title='Unemployment Rate'
    )

    fig.update_layout(
        height=600
    )

    return fig
