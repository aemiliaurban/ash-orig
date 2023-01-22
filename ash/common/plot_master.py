from unittest.mock import patch

import pandas as pd
import plotly.express as px
import streamlit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from ash.common.plotly_modified_dendrogram import create_dendrogram_modified


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

    def plot_selected_features_streamlit(self):
        desired_columns = streamlit.multiselect(
            "Choose 2 features to plot.", self.input_data.columns
        )
        if len(desired_columns) != 2:
            streamlit.write("Please choose 2 features to plot.")
        else:
            to_plot = self.input_data[desired_columns]
            fig = px.scatter(
                to_plot,
                x="Murder",
                y="Assault",
                color=self.labels,
                labels={"color": "states"},
            )
            return fig

    def plot_custom_dendrogram(self, input_data, color_threshold):
        with patch(
            "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
            new=create_dendrogram_modified,
        ) as create_dendrogram:
            fig = self.plot_interactive(create_dendrogram, input_data, color_threshold)
            return fig
