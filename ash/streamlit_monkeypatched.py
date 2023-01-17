import math
from unittest.mock import patch

import pandas as pd
import streamlit
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go

from common.data_parser import RDataParser
from common.input_data import INPUT_DATA_DENDROGRAM, US_ARRESTS
from common.plotly_modified_dendrogram import create_dendrogram_modified


def plot_interactive(func, data, threshold, labels):
    return func(
        data,
        color_threshold=threshold,
        labels=labels
    )


def df_to_plotly(df: pd.DataFrame, desired_columns: list):
    df = df[desired_columns]
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}


r = RDataParser(INPUT_DATA_DENDROGRAM)
r.convert_merge_matrix()
r.add_joining_height()
dendrogram(r.merge_matrix)

streamlit.header("Cluster Analysis")

color_threshold = streamlit.slider("Select color threshold.", 0, math.ceil(max(r.joining_height)))

with patch(
    "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
    new=create_dendrogram_modified,
) as create_dendrogram:
    fig = plot_interactive(create_dendrogram, r.merge_matrix, color_threshold, r.labels)

streamlit.plotly_chart(fig)

desired_columns = streamlit.multiselect("What data would you like to plot in a heatmap.", list(US_ARRESTS.columns))

fig_heatmap = go.Figure(data=go.Heatmap(df_to_plotly(US_ARRESTS, desired_columns=desired_columns)))

streamlit.plotly_chart(fig_heatmap)