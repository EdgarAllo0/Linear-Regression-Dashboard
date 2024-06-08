# Libraries
import streamlit as st
import plotly.graph_objects as go
import seaborn as sns


@st.cache_resource
def BetasPlot(
        betas_list: list,
        standard_errors_list: list
):
    betas_names = [f'b{i}' for i in range(len(betas_list))]

    # Colors
    palette = list(sns.color_palette("Spectral", len(betas_list)).as_hex())

    # Create plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=betas_names,
            y=betas_list,
            error_y=dict(type='data', array=standard_errors_list, visible=True),
            mode='markers',
            marker=dict(
                size=10,
                symbol='circle',
                color=palette
            ),
            name='Betas'
        )
    )

    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(betas_list) - 0.5,
        y1=0,
        line=dict(
            color="black",
            width=2,
            dash="dash"  # Línea punteada
        )
    )

    # Config
    fig.update_layout(
        title='Betas Graph with Standard Errors',
        xaxis_title='Betas',
        yaxis_title='Values',
        yaxis=dict(tickformat='.2f')
    )

    fig.update_layout(
        height=400
    )

    return fig
