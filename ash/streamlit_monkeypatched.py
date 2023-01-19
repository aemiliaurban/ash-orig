import math
from unittest.mock import patch

import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
import streamlit
from common.data_parser import RDataParser
from common.input_data import INPUT_DATA_DENDROGRAM, US_ARRESTS
from common.plotly_modified_dendrogram import create_dendrogram_modified
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


class PlotMaster:
    def __init__(self, input_data, labels: list[str], order: list[int | float]):
        self.input_data = input_data
        self.labels = labels
        self.order = order

    def plot_interactive(self, func, data, threshold):
        return func(data, color_threshold=threshold, labels=self.labels)

    def order_labels(self):
        ordered_labels = []
        for index in self.order:
            ordered_labels.append(self.labels[int(index)])
        return ordered_labels

    def df_to_plotly(
        self,
        df: pd.DataFrame,
        desired_columns: list,
    ):
        df = df.reindex(self.order)
        df = df[desired_columns].T
        return {"z": df.values, "x": self.order_labels(), "y": df.index.tolist()}

    def plot_pca(self):
        pca = PCA(2).fit_transform(self.input_data)
        fig = px.scatter(pca, x=0, y=1, color=self.labels)
        return fig

    def plot_pca_3d(self):
        pca = PCA(3).fit_transform(self.input_data)
        fig = px.scatter_3d(pca, x=0, y=1, z=2, color=self.labels)
        return fig

    def plot_all_dimensions(self):
        features = self.input_data.columns
        fig = px.scatter_matrix(self.input_data, dimensions=features, color=self.labels)
        fig.update_traces(diagonal_visible=True)
        return fig

    def plot_tsne(self):
        tsne = TSNE(n_components=2, random_state=0, perplexity=5).fit_transform(
            self.input_data
        )
        fig = px.scatter(tsne, x=0, y=1, color=self.labels, labels={"color": "states"})
        return fig

    def plot_tsne_3D(self):
        tsne = TSNE(n_components=3, random_state=0, perplexity=5).fit_transform(
            self.input_data
        )
        fig = px.scatter_3d(
            tsne, x=0, y=1, z=2, color=self.labels, labels={"color": "states"}
        )
        return fig

    def plot_umap(self):
        umap = UMAP(n_components=2, init="random", random_state=0).fit_transform(
            self.input_data
        )
        fig = px.scatter(umap, x=0, y=1, color=self.labels, labels={"color": "states"})
        return fig

    def plot_umap_3D(self):
        umap = UMAP(n_components=3, init="random", random_state=0).fit_transform(
            self.input_data
        )
        fig = px.scatter_3d(
            umap, x=0, y=1, z=2, color=self.labels, labels={"color": "states"}
        )
        return fig

    def plot_selected_features(self):
        desired_columns = streamlit.multiselect(
            "Choose 2 features to plot.", self.input_data.columns
        )
        if len(desired_columns) != 2:
            streamlit.write("Please choose 2 features to plot.")
        else:
            to_plot = self.input_data[desired_columns]
            print(to_plot)
            fig = px.scatter(
                to_plot,
                x="Murder",
                y="Assault",
                color=self.labels,
                labels={"color": "states"},
            )
            return fig


def plot_input_data_reduced(plot_input_data: list[str]):
    if "Select two features" in plot_input_data:
        two_features_plot = plot_master.plot_selected_features()
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
dendrogram(r.merge_matrix)

streamlit.header("Cluster Analysis")

color_threshold = streamlit.slider(
    "Select color threshold.", 0, math.ceil(max(r.joining_height))
)
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
