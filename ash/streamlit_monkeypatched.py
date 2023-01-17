import math
from unittest.mock import patch

import pandas as pd
import plotly
import plotly.graph_objects as go
import streamlit
from common.data_parser import RDataParser
from common.input_data import INPUT_DATA_DENDROGRAM, US_ARRESTS
from common.plotly_modified_dendrogram import create_dendrogram_modified
from scipy.cluster.hierarchy import dendrogram


def plot_interactive(func, data, threshold, labels):
    return func(data, color_threshold=threshold, labels=labels)


def order_labels(order: list[int | float], labels: list[str]):
    ordered_labels = []
    for index in order:
        ordered_labels.append(labels[int(index)])
    return ordered_labels


def df_to_plotly(df: pd.DataFrame, desired_columns: list, order: list[int | float], labels: list[str]):
    df = df.reindex(order)
    df = df[desired_columns].T
    return {"z": df.values, "x": order_labels(order, labels), "y": df.index.tolist()}


r = RDataParser(INPUT_DATA_DENDROGRAM)
r.convert_merge_matrix()
r.add_joining_height()
dendrogram(r.merge_matrix)

streamlit.header("Cluster Analysis")

color_threshold = streamlit.slider(
    "Select color threshold.", 0, math.ceil(max(r.joining_height))
)
desired_columns = streamlit.multiselect(
    "What data would you like to plot in a heatmap.", list(US_ARRESTS.columns)
)

with patch(
    "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
    new=create_dendrogram_modified,
) as create_dendrogram:
    fig = plot_interactive(create_dendrogram, r.merge_matrix, color_threshold, r.labels)

streamlit.plotly_chart(fig)

plotly.graph_objs.Layout(yaxis=dict(autorange="reversed"))
fig_heatmap = go.Figure(
    data=go.Heatmap(df_to_plotly(US_ARRESTS, desired_columns, r.order, r.labels))
)

streamlit.plotly_chart(fig_heatmap)
