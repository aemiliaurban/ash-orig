from unittest.mock import patch

from dash import Dash, Input, Output, dcc, html

from ash.common.data_parser import RDataParser
from ash.common.input_data import INPUT_DATA_DENDROGRAM, US_ARRESTS
from ash.common.plot_master import PlotMaster
from ash.common.plotly_modified_dendrogram import create_dendrogram_modified

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
        #dcc.Graph(figure=plot_master.plot_custom_dendrogram(r.merge_matrix, value))
        html.Div(id="slider-output-container"),
    ]
)


@app.callback(
    Output("slider-output-container", "children"),
    Input("color-threshold-slider", "value"),
)
def update_output(value):
    with patch(
            "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
            new=create_dendrogram_modified,
    ) as create_dendrogram:
        fig = plot_master.plot_interactive(
            create_dendrogram, r.merge_matrix, value
        )
    return f"You have selected {value}"


if __name__ == "__main__":
    app.run_server(debug=True)
