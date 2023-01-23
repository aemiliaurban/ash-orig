from unittest.mock import patch

import plotly
import plotly.graph_objects as go
import streamlit
from common.data_parser import RDataParser
from common.input_data import INPUT_DATA_DENDROGRAM, US_ARRESTS
from common.plot_master import PlotMaster
from common.plotly_modified_dendrogram import create_dendrogram_modified
from scipy.cluster.hierarchy import dendrogram


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
        pca = plot_master.plot_pca_3d()
        streamlit.plotly_chart(pca)
    if "tSNE" in plot_input_data:
        tsne = plot_master.plot_tsne()
        streamlit.plotly_chart(tsne)
    if "tSNE_3D" in plot_input_data:
        tsne = plot_master.plot_tsne_3D()
        streamlit.plotly_chart(tsne)
    if "UMAP" in plot_input_data:
        umap = plot_master.plot_umap()
        streamlit.plotly_chart(umap)
    if "UMAP_3D" in plot_input_data:
        umap = plot_master.plot_umap_3D()
        streamlit.plotly_chart(umap)


r = RDataParser(INPUT_DATA_DENDROGRAM)
r.convert_merge_matrix()
r.add_joining_height()

streamlit.header("Cluster Analysis")

color_threshold = streamlit.slider("Select color threshold.", 0, r.max_tree_height)
desired_columns = streamlit.multiselect(
    "What data would you like to plot in a heatmap.", list(US_ARRESTS.columns)
)

plot_master = PlotMaster(US_ARRESTS, r.labels, r.order)

with patch(
    "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
    new=create_dendrogram_modified,
) as create_dendrogram:
    fig = plot_master.plot_interactive(
        create_dendrogram, r.merge_matrix, color_threshold
    )

streamlit.plotly_chart(fig)

plotly.graph_objs.Layout(yaxis=dict(autorange="reversed"))
fig_heatmap = go.Figure(
    data=go.Heatmap(plot_master.df_to_plotly(US_ARRESTS, desired_columns))
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
