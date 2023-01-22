from unittest.mock import patch
import plotly.graph_objects as go

import matplotlib
from dash import Dash, Input, Output, dcc, html

from ash.common.data_parser import RDataParser
from ash.common.input_data import INPUT_DATA_DENDROGRAM, US_ARRESTS
from ash.common.plot_master import PlotMaster
from ash.common.plotly_modified_dendrogram import create_dendrogram_modified

matplotlib.pyplot.switch_backend("Agg")

r = RDataParser(INPUT_DATA_DENDROGRAM)
r.convert_merge_matrix()
r.add_joining_height()

plot_master = PlotMaster(US_ARRESTS, r.labels, r.order)

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
        dcc.Graph(id="dendrogram-graph"),
        dcc.Graph(id="heatmap-graph"),
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
            multi=True,
            id="plot_dropdown",
        ),
    ]
)


@app.callback(
    Output("dendrogram-graph", "figure"),
    Input("color-threshold-slider", "value"),
)
def update_output(value):
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
def update_output(value):
    print(value)
    while type(value) != list or len(value) != 2:
        pass
    fig_heatmap = go.Figure(
        data=go.Heatmap(plot_master.df_to_plotly(US_ARRESTS, value))
    )
    return fig_heatmap


if __name__ == "__main__":
    app.run_server(debug=True)
