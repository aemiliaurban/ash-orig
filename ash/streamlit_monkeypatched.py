import math
from unittest.mock import patch

import streamlit
from scipy.cluster.hierarchy import dendrogram

from common.data_parser import RDataParser
from common.input_data import INPUT_DATA
from common.plotly_modified_dendrogram import create_dendrogram_modified


def plot_interactive(func, data, threshold):
    return func(
        data,
        color_threshold=threshold,
    )


r = RDataParser(INPUT_DATA)
r.convert_merge_matrix()
r.add_joining_height()
dendrogram(r.merge_matrix)

streamlit.header("Cluster Analysis")

color_threshold = streamlit.slider("Select color threshold.", 0, math.ceil(max(r.joining_height)))

with patch(
    "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
    new=create_dendrogram_modified,
) as create_dendrogram:
    fig = plot_interactive(create_dendrogram, r.merge_matrix, color_threshold)

streamlit.plotly_chart(fig)
