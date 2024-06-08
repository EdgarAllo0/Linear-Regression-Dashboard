# Libraries
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


@st.cache_resource
def SSTreemap(
        SSR: float,
        SSE: float,
):

    SST = SSR + SSE

    # Create the Treemap
    fig = go.Figure(
        go.Treemap(
            labels=["SSR", "SSE"],
            parents=[f"SST {SST.round(4)}", f"SST {SST.round(4)}"],
            values=[SSR, SSE],
            textinfo="label+value+percent parent",
            marker_colorscale='Blues'
        )
    )

    # Config
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.update_layout(title="Treemap of the Squared Sums")
    fig.update_layout(treemapcolorway=["#636EFA", "#EF553B", "#00CC96"])

    return fig
