import timeit

import dash
import dash_bio as dashbio
import plotly.figure_factory
import scipy
import streamlit_app
from dash import dcc, html
from dash.dependencies import Input, Output
from hierarchical_clustering import example, iris_example

X, labels, Z, dendrogram = example()

iris, iris_data, agglomerative_clustering, iris_dendrogram = iris_example()


app = dash.Dash(__name__)


def function():
    df = iris

    columns = list(df.columns.values)
    rows = list(df.index)

    app.layout = html.Div(
        [
            "Rows to display",
            dcc.Dropdown(
                id="choose_dataset_attributes",
                options=[{"label": column, "value": column} for column in columns],
                value=columns,
                multi=True,
            ),
            "Choose distance metric",
            dcc.Dropdown(
                id="choose_distance_metric",
                options=[
                    "euclidean",
                    "minkowski",
                    "cityblock",
                    "seuclidean",
                    "sqeuclidean",
                    "cosine",
                    "correlation",
                    "hamming",
                    "jaccard",
                    "jensenshannon",
                    "chebyshev",
                    "canberra",
                    "braycurtis",
                    "mahalanobis",
                    "yule",
                    "matching",
                    "dice",
                    "kulczynski1",
                    "rogerstanimoto",
                    "russellrao",
                    "sokalmichener",
                    "sokalsneath",
                ],
                value="euclidean",
                multi=False,
            ),
            html.Div(id="my-default-clustergram"),
        ]
    )

    @app.callback(
        Output("my-default-clustergram", "children"),
        Input("choose_dataset_attributes", "value"),
        Input("choose_distance_metric", "value"),
    )
    def update_clustergram(columns, distance_metric):
        if len(columns) < 2:
            return "Please select at least two rows to display."

        return dcc.Graph(
            figure=dashbio.Clustergram(
                data=df[columns],
                row_dist=distance_metric,
                column_labels=columns,
                row_labels=rows,
                color_threshold={"row": 250, "col": 700},
                hidden_labels="row",
                height=1000,
                width=800,
            )
        )


if __name__ == "__main__":
    # print(timeit.timeit("function()", globals=globals()))
    function()
    app.run_server(debug=True)
