import os

import numpy
import pandas as pd
import plotly.express as px
import streamlit
from plotly.graph_objs import graph_objs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from .input_data import DATA_FOLDER

REDUCED_DIMENSIONS_FOLDER = "reduced_dimensions"

numpy.set_printoptions(threshold=9999999999999)


class PlotMaster:
    def __init__(
        self, input_data, labels: list[str], order: list[int | float], color_map: dict
    ):
        self.input_data = input_data
        self.labels = labels
        self.order = order
        self.color_map = color_map

    def plot_interactive(self, data, layout):
        return graph_objs.Figure(data=data, layout=layout)

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
        pca = self.read_reduction("pca.txt", REDUCED_DIMENSIONS_FOLDER)
        if pca:
            fig = px.scatter(
                pca, x=0, y=1, color=self.color_map, hover_name=self.labels, title="PCA"
            )
        else:
            pca = PCA(2).fit_transform(self.input_data)
            print(f"{pca=}")
            self.save_reduction(pca, "pca.txt", REDUCED_DIMENSIONS_FOLDER)
            fig = px.scatter(
                pca, x=0, y=1, color=self.color_map, hover_name=self.labels, title="PCA"
            )
        return fig

    def plot_pca_3d(self):
        pca = self.read_reduction("pca_3D.txt", REDUCED_DIMENSIONS_FOLDER)
        if pca:
            fig = px.scatter_3d(
            pca, x=0, y=1, z=2, color=self.color_map, hover_name=self.labels, title="PCA 3D"
        )
        else:
            pca = PCA(3).fit_transform(self.input_data)
            self.save_reduction(pca, "pca_3D.txt", REDUCED_DIMENSIONS_FOLDER)
            fig = px.scatter_3d(
            pca, x=0, y=1, z=2, color=self.color_map, hover_name=self.labels, title="PCA 3D"
        )
        return fig

    def plot_all_dimensions(self):
        features = self.input_data.columns
        fig = px.scatter_matrix(
            self.input_data,
            dimensions=features,
            color=self.color_map,
            hover_name=self.labels,
        )
        fig.update_traces(diagonal_visible=True)
        return fig

    def plot_tsne(self):
        tsne = self.read_reduction("tsne.txt", REDUCED_DIMENSIONS_FOLDER)
        if tsne:
            fig = px.scatter(
                tsne,
                x=0,
                y=1,
                color=self.color_map,
                hover_name=self.labels,
                title="TSNE",
            )
        else:
            tsne = TSNE(n_components=2, random_state=0, perplexity=5).fit_transform(
                self.input_data
            )
            self.save_reduction(tsne, "tsne.txt", REDUCED_DIMENSIONS_FOLDER)
            fig = px.scatter(
                tsne,
                x=0,
                y=1,
                color=self.color_map,
                hover_name=self.labels,
                title="TSNE",
            )
        return fig

    def plot_tsne_3D(self):
        tsne = self.read_reduction("tsne_3D.txt", REDUCED_DIMENSIONS_FOLDER)
        if tsne:
            fig = px.scatter(
                tsne,
                x=0,
                y=1,
                color=self.color_map,
                hover_name=self.labels,
                title="TSNE",
            )
        else:
            tsne = TSNE(n_components=3, random_state=0, perplexity=5).fit_transform(
                self.input_data
            )
            self.save_reduction(tsne, "tsne_3D.txt", REDUCED_DIMENSIONS_FOLDER)
            fig = px.scatter(
                tsne,
                x=0,
                y=1,
                color=self.color_map,
                hover_name=self.labels,
                title="TSNE",
            )
        return fig

    def plot_umap(self):
        umap = self.read_reduction("umap.txt", REDUCED_DIMENSIONS_FOLDER)
        if umap:
            fig = px.scatter(
                umap,
                x=0,
                y=1,
                color=self.color_map,
                hover_name=self.labels,
                title="UMAP",
            )
        else:
            umap = UMAP(n_components=2, init="random", random_state=0).fit_transform(
                self.input_data
            )
            self.save_reduction(umap, "umap.txt", REDUCED_DIMENSIONS_FOLDER)
            fig = px.scatter(
                umap,
                x=0,
                y=1,
                color=self.color_map,
                hover_name=self.labels,
                title="UMAP",
            )
        return fig

    def plot_umap_3D(self):
        umap = self.read_reduction("umap.txt", REDUCED_DIMENSIONS_FOLDER)
        if umap:
            fig = px.scatter(
                umap,
                x=0,
                y=1,
                color=self.color_map,
                hover_name=self.labels,
                title="UMAP",
            )
        else:
            umap = UMAP(n_components=3, init="random", random_state=0).fit_transform(
                self.input_data
            )
            self.save_reduction(umap, "umap.txt", REDUCED_DIMENSIONS_FOLDER)
            fig = px.scatter(
                umap,
                x=0,
                y=1,
                color=self.color_map,
                hover_name=self.labels,
                title="UMAP",
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
                x=to_plot.columns[0],
                y=to_plot.columns[1],
                color=self.color_map,
                hover_name=self.labels,
                labels={"color": "states"},
            )
            return fig

    def plot_selected_features(self, desired_columns):
        to_plot = self.input_data[desired_columns]
        fig = px.scatter(
            to_plot,
            x=to_plot.columns[0],
            y=to_plot.columns[1],
            color=self.color_map,
            hover_name=self.labels,
            labels={"color": "states"},
        )
        return fig

    @staticmethod
    def save_reduction(
        data, filename: str, subfolder: str, path_to_folder: str = DATA_FOLDER
    ) -> None:
        with open(os.path.join(path_to_folder, subfolder, filename), "w") as file:
            file.write(str(data))

    @staticmethod
    def read_reduction(
        filename: str, subfolder: str, path_to_folder: str = DATA_FOLDER
    ) -> list[list[float]] | None:
        data = []
        try:
            with open(os.path.join(path_to_folder, subfolder, filename), "r") as file:
                for line in file.readlines():
                    l = list(
                        map(
                            float,
                            line.strip().replace("[", "").replace("]", "").split(),
                        )
                    )
                    data.append(l)
        except:
            return None
        return data

