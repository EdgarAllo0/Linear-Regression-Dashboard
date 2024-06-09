import streamlit as st
import numpy as np
import seaborn as sns
import random
from scipy.stats import f
import plotly.graph_objects as go


@st.cache_resource
def FisherTest(
        f_stat: float,
        df_num: int,
        df_den: int,
):
    # Colors:
    palette = list(sns.color_palette("Spectral", 20).as_hex())

    color1 = random.choice(palette)
    color2 = random.choice(palette)

    # F-distribution
    x = np.linspace(0, 10, 1000)

    y = f.pdf(x, df_num, df_den)

    # Level of Significance
    alpha = 0.05

    # Critical F-value
    critical_value = f.ppf(1 - alpha / 2, df_den, df_num)

    p_value = 2 * (f.sf(
        f_stat,
        df_num, df_den
    ).round(3)
                   )

    # Now create the figure
    fig = go.Figure()

    # Add the F-distribution
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color=color2, width=2),
            name=f'F-distribution (df_num={df_num}, df_den={df_den})'
        )
    )

    # Add reject area
    x_fill_right = np.linspace(critical_value, 50, 100)
    y_fill_right = f.pdf(x_fill_right, df_num, df_den)

    fig.add_trace(
        go.Scatter(
            x=np.concatenate(
                [x_fill_right, x_fill_right[::-1]
                 ]
            ),
            y=np.concatenate(
                [y_fill_right, np.zeros_like(y_fill_right)
                 ]
            ),
            fill='toself',
            fillcolor=color1,
            line=dict(color=color1),
            name='Reject Area'
        )
    )

    # Add vertical line on critical value
    fig.add_trace(
        go.Scatter(
            x=[critical_value, critical_value],
            y=[0, f.pdf(critical_value, df_num, df_den)],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name=f'Critical Value {critical_value:.2f}'
        )
    )

    # Add the hypothesis F-value
    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=f_stat,
            y0=0,
            x1=f_stat,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color='#3E5F8A',
                width=2,
                dash="dash"
            ),
            name=f'F-Value: {f_stat}'
        )
    )

    fig.update_layout(
        shapes=[
            dict(
                type='line',
                x0=f_stat,
                x1=f_stat,
                y0=0,
                y1=1,
                xref='x',
                yref='paper',
                line=dict(
                    color='#3E5F8A',
                    width=2,
                    dash="dash"
                )
            )
        ],
        annotations=[
            dict(
                x=f_stat,
                y=1,
                xref='x',
                yref='paper',
                text=f'F-value: {f_stat} (p-value: {p_value:.2f})',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40
            )
        ]
    )

    # Config
    fig.update_layout(
        title=f'Fisher Distribution with df_num={df_num}, df_den={df_den}',
        xaxis_title='F-values',
        yaxis_title='Density',
        showlegend=True
    )

    fig.update_layout(
        height=400
    )

    return fig
