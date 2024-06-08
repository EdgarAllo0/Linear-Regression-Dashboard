# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def calculate_sse(
        x_series: pd.Series,
        y_series: pd.Series,
        b0: int,
        b1: int,
):
    y_pred = b0 + b1 * x_series

    errors = y_series - y_pred

    sse = np.sum(errors ** 2)

    return sse


@st.cache_resource
def SSEPlot(
        x_series: pd.Series,
        y_series: pd.Series,
        b0_input: int,
        b1_input: int,
):
    # Range for the betas
    b0_range = np.linspace(-10, 10, 500)
    b1_range = np.linspace(-1, 1, 50)
    B0, B1 = np.meshgrid(b0_range, b1_range)

    # Calculate SSE for each pair of betas (b0, b1)
    SSE = np.array(
        [
            [
                calculate_sse(
                    x_series,
                    y_series,
                    b0,
                    b1,
                ) for b0 in b0_range
            ] for b1 in b1_range
        ]
    )

    # Coefficient to highlight
    sse_example = calculate_sse(
        x_series,
        y_series,
        b0_input,
        b1_input
    )

    sse_example = max(sse_example, 1e-6)

    # Create the Surface plot
    surface = go.Surface(
        z=SSE,
        x=B0,
        y=B1,
        colorscale='Spectral'
    )

    # Highlight the Preselected coefficients
    point = go.Scatter3d(
        x=[b0_input],
        y=[b1_input],
        z=[sse_example],
        mode='markers',
        marker=dict(size=5, color='red'),
        name=f'b0={b0_input}, '
             f'b1={b1_input}, '
             f'SSE={sse_example:.2f}'
    )

    # Define Figure
    fig = go.Figure(data=[surface, point])

    annotation_text = f"SSR = {sse_example:.2f}"
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.95, y=0.95,
        text=annotation_text,
        showarrow=False,
        bordercolor="black",
        borderwidth=1
    )

    # Config
    fig.update_layout(
        title='Squared Sum of Residuals Plot',
        scene=dict(
            xaxis_title='b0',
            yaxis_title='b1',
            zaxis_title='SSR'
        )
    )

    fig.update_layout(
        height=600
    )

    return fig
