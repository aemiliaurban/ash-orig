from unittest.mock import patch

import matplotlib
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from ash.common.data_parser import RDataParser
from ash.common.input_data import INPUT_DATA_DENDROGRAM, US_ARRESTS
from ash.common.plot_master import PlotMaster
from ash.common.plotly_modified_dendrogram import create_dendrogram_modified
from plotly.graph_objs import graph_objs

matplotlib.pyplot.switch_backend("agg")

r = RDataParser(INPUT_DATA_DENDROGRAM)
r.convert_merge_matrix()
r.add_joining_height()


def plot_input_data_reduced(plot_input_data: str, plot_master: PlotMaster):
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
        dcc.Dropdown(
            [
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
        dcc.Dropdown(
            list(US_ARRESTS.columns), multi=True, id="dropdown-selected-features-plot"
        ),
        dcc.Graph(id="two-features", figure=go.Figure()),
        dcc.Graph(id="heatmap-graph", figure=go.Figure()),
        dcc.Graph(id="reduced-graph", figure=go.Figure()),
        dcc.Store(id="dendrogram_memory"),
    ]
)


@app.callback(
    Output("dendrogram_memory", "data"),
    Input("color-threshold-slider", "value"),
)
def create_dendrogram(value):
    with patch(
        "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
        new=create_dendrogram_modified,
    ) as create_dendrogram:
        custom_dendrogram = create_dendrogram(
            r.merge_matrix, color_threshold=value, labels=r.labels
        )
        to_return = {
            "leaves_color_map_translated": custom_dendrogram.leaves_color_map_translated,
            "labels": custom_dendrogram.labels,
            "data": custom_dendrogram.data,
            "layout": custom_dendrogram.layout,
        }
        return to_return


@app.callback(
    Output("dendrogram-graph", "figure"),
    Input("dendrogram_memory", "data"),
)
def plot_dendrogram(data):
    fig = graph_objs.Figure(data=data["data"], layout=data["layout"])
    return fig


@app.callback(
    Output("heatmap-graph", "figure"),
    Input("dropdown-heatmap-plot", "value"),
    Input("dendrogram_memory", "data"),
)
def plot_heatmap(value, data):
    plot_master = PlotMaster(
        US_ARRESTS, data["labels"], r.order, data["leaves_color_map_translated"]
    )
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
    Output("two-features", "figure"),
    Input("dropdown-selected-features-plot", "value"),
    Input("dendrogram_memory", "data"),
)
def plot_two_selected_features(value, data):
    plot_master = PlotMaster(
        US_ARRESTS, data["labels"], r.order, data["leaves_color_map_translated"]
    )
    if type(value) != list or len(value) != 2:
        feature_plot = go.Figure(
            plot_master.plot_selected_features(US_ARRESTS.columns[0:2])
        )
    else:
        feature_plot = go.Figure(plot_master.plot_selected_features(value))
    return feature_plot


@app.callback(
    Output("reduced-graph", "figure"),
    Input("plot_dropdown", "value"),
    Input("dendrogram_memory", "data"),
)
def plot_data_reduced(value, data):
    plot_master = PlotMaster(
        US_ARRESTS, data["labels"], r.order, data["leaves_color_map_translated"]
    )
    return plot_input_data_reduced(value, plot_master)


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_silence_routes_logging=False)
