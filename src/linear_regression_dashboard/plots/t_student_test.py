import streamlit as st
import numpy as np
import seaborn as sns
import random
from scipy.stats import t
import plotly.graph_objects as go


@st.cache_resource
def TStudentTest(
        t_stat: float,
        degrees_of_freedom: int,
):
    # Colors:
    palette = list(sns.color_palette("Spectral", 20).as_hex())

    color1 = random.choice(palette)
    color2 = random.choice(palette)

    # t-student Distribution
    x = np.linspace(-10, 10, 1000)

    y = t.pdf(x, degrees_of_freedom)

    # Level of Significance
    alpha = 0.05

    # Critical t-value (tends to 1.96)
    critical_value = t.ppf(1 - alpha / 2, degrees_of_freedom)

    p_value = 2*(t.sf(
        abs(t_stat),
        degrees_of_freedom
    ).round(3)
                 )

    # Now create the figure
    fig = go.Figure()

    # Add the t-student distribution
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color=color2, width=2),
            name=f't-distribution df={degrees_of_freedom}'
        )
    )

    # Add reject areas
    x_fill_left = np.linspace(-5, -critical_value, 100)
    y_fill_left = t.pdf(x_fill_left, degrees_of_freedom)

    fig.add_trace(
        go.Scatter(
            x=np.concatenate(
                [x_fill_left, x_fill_left[::-1]
                 ]
            ),
            y=np.concatenate(
                [y_fill_left, np.zeros_like(y_fill_left)
                 ]
            ),
            fill='toself',
            fillcolor=color1,
            line=dict(color=color1),
            name='Reject Area'
        )
    )

    x_fill_right = np.linspace(critical_value, 5, 100)
    y_fill_right = t.pdf(x_fill_right, degrees_of_freedom)

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

    # Add vertical lines on critical values
    fig.add_trace(
        go.Scatter(
            x=[-critical_value, -critical_value],
            y=[0, t.pdf(-critical_value, degrees_of_freedom)],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name=f'Left Critical Value -{critical_value:.2f}'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[critical_value, critical_value],
            y=[0, t.pdf(critical_value, degrees_of_freedom)],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name=f'Right Critical Value {critical_value:.2f}'
        )
    )

    # Add the hypothesis t-value
    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=t_stat,
            y0=0,
            x1=t_stat,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color='#3E5F8A',
                width=2,
                dash="dash"
            ),
            name=f'T-Value: {t_stat}'
        )
    )

    fig.update_layout(
        shapes=[
            dict(
                type='line',
                x0=t_stat,
                x1=t_stat,
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
                x=t_stat,
                y=1,
                xref='x',
                yref='paper',
                text=f't-value: {t_stat} (p-value: {p_value:.2f})',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40
            )
        ]
    )

    # Config
    fig.update_layout(
        title=f'T-Student Distribution with {degrees_of_freedom} degrees of freedom',
        xaxis_title='t-values',
        yaxis_title='Density',
        showlegend=True
    )

    fig.update_layout(
        height=400
    )

    return fig
