from unittest.mock import patch

import plotly.graph_objects as go
import streamlit
from common.data_parser import RDataParser
from common.input_data import FLOW_CYTOMETRY, INPUT_FLOW_DATA_DENDROGRAM
from common.plot_master import PlotMaster
from common.plotly_modified_dendrogram import create_dendrogram_modified
from plotly.graph_objs import graph_objs


def plot_input_data_reduced(plot_input_data: list[str]):
    if "Select two features" in plot_input_data:
        two_features_plot = plot_master.plot_selected_features_streamlit()
        if two_features_plot:
            streamlit.plotly_chart(two_features_plot)
    if "All dimensions" in plot_input_data:
        all_dimensions = plot_master.plot_all_dimensions()
        streamlit.plotly_chart(all_dimensions)
    if "PCA" in plot_input_data:
        pca = plot_master.plot_pca()
        streamlit.plotly_chart(pca)
    if "PCA_3D" in plot_input_data:
        pca = plot_master.plot_pca(dimensions=3)
        streamlit.plotly_chart(pca)
    if "tSNE" in plot_input_data:
        tsne = plot_master.plot_tsne()
        streamlit.plotly_chart(tsne)
    if "tSNE_3D" in plot_input_data:
        tsne = plot_master.plot_tsne(dimensions=3)
        streamlit.plotly_chart(tsne)
    if "UMAP" in plot_input_data:
        umap = plot_master.plot_umap()
        streamlit.plotly_chart(umap)
    if "UMAP_3D" in plot_input_data:
        umap = plot_master.plot_umap(dimensions=3)
        streamlit.plotly_chart(umap)


r = RDataParser(INPUT_FLOW_DATA_DENDROGRAM)
r.convert_merge_matrix()
r.add_joining_height()

streamlit.header("Cluster Analysis")

color_threshold = streamlit.slider("Select color threshold.", 0, r.max_tree_height)
desired_columns = streamlit.multiselect(
    "What data would you like to plot in a heatmap.", list(FLOW_CYTOMETRY.columns)
)


def compute_dendrogram_fig():
    with patch(
        "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
        new=create_dendrogram_modified,
    ) as create_dendrogram:
        custom_dendrogram = create_dendrogram(
            r.merge_matrix, color_threshold=color_threshold, labels=r.labels
        )
        custom_dendrogram_color_map = custom_dendrogram.leaves_color_map_translated
        fig = graph_objs.Figure(
            data=custom_dendrogram.data, layout=custom_dendrogram.layout
        )
    return fig, custom_dendrogram, custom_dendrogram_color_map


fig, custom_dendrogram, custom_dendrogram_color_map = compute_dendrogram_fig()

streamlit.plotly_chart(fig)

plot_master = PlotMaster(
    FLOW_CYTOMETRY, custom_dendrogram.labels, r.order, custom_dendrogram_color_map
)

fig_heatmap = go.Figure(
    data=go.Heatmap(plot_master.df_to_plotly(FLOW_CYTOMETRY, desired_columns))
)

streamlit.plotly_chart(fig_heatmap)

plot_input_data = streamlit.multiselect(
    "How would you like to plot the input data.",
    [
        "Select two features",
        "All dimensions",
        "PCA",
        "PCA_3D",
        "tSNE",
        "tSNE_3D",
        "UMAP",
        "UMAP_3D",
    ],
)

plot_input_data_reduced(plot_input_data)
