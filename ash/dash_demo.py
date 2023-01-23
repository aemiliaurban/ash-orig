from unittest.mock import patch

import matplotlib
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from ash.common.data_parser import RDataParser
from ash.common.input_data import INPUT_DATA_DENDROGRAM, US_ARRESTS
from ash.common.plot_master import PlotMaster
from ash.common.plotly_modified_dendrogram import create_dendrogram_modified

matplotlib.pyplot.switch_backend("agg")

r = RDataParser(INPUT_DATA_DENDROGRAM)
r.convert_merge_matrix()
r.add_joining_height()

plot_master = PlotMaster(US_ARRESTS, r.labels, r.order)


def plot_input_data_reduced(plot_input_data: str):
    # if "Select two features" in plot_input_data:
    #     return plot_master.plot_selected_features_streamlit()
    if plot_input_data == "All dimensions":
        return plot_master.plot_all_dimensions()
    elif plot_input_data == "PCA":
        return plot_master.plot_pca()
    elif "PCA_3D" in plot_input_data:
        return plot_master.plot_pca_3d()
    elif plot_input_data == "tSNE":
        return plot_master.plot_tsne()
    elif plot_input_data == "tSNE_3D":
        return plot_master.plot_tsne_3D()
    elif plot_input_data == "UMAP":
        return plot_master.plot_umap()
    elif plot_input_data == "UMAP_3D":
        return plot_master.plot_umap_3D()
    else:
        return plot_master.plot_pca()


app = Dash(__name__)
app.layout = html.Div(
    [
        dcc.Slider(
            min=0,
            max=r.max_tree_height,
            value=10,
            marks=r.height_marks,
            id="color-threshold-slider",
        ),
        dcc.Dropdown(list(US_ARRESTS.columns), multi=True, id="dropdown-heatmap-plot"),
        dcc.Graph(id="dendrogram-graph", figure=go.Figure()),
        dcc.Graph(id="heatmap-graph", figure=go.Figure()),
        dcc.Dropdown(
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
            id="plot_dropdown",
            value=["PCA"],
        ),
        dcc.Graph(id="reduced-graph", figure=go.Figure()),
    ]
)


@app.callback(
    Output("dendrogram-graph", "figure"),
    Input("color-threshold-slider", "value"),
)
def create_dendrogram(value):
    with patch(
        "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
        new=create_dendrogram_modified,
    ) as create_dendrogram:
        fig = plot_master.plot_interactive(create_dendrogram, r.merge_matrix, value)
    return fig


@app.callback(
    Output("heatmap-graph", "figure"),
    Input("dropdown-heatmap-plot", "value"),
)
def plot_heatmap(value):
    if type(value) != list or len(value) != 2:
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                plot_master.df_to_plotly(US_ARRESTS, US_ARRESTS.columns[0:2])
            )
        )
    else:
        fig_heatmap = go.Figure(
            data=go.Heatmap(plot_master.df_to_plotly(US_ARRESTS, value))
        )
    return fig_heatmap


@app.callback(
    Output("reduced-graph", "figure"),
    Input("plot_dropdown", "value"),
)
def plot_data_reduced(value):
    return plot_input_data_reduced(value)


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_silence_routes_logging=False)
