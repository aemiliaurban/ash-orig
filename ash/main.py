import plotly.figure_factory
import streamlit
from hierarchical_clustering import example, iris_example

import dash
from dash.dependencies import Input, Output
import dash_bio as dashbio
from dash import html, dcc

X, labels, Z, dendrogram = example()

iris, iris_data, agglomerative_clustering, iris_dendrogram = iris_example()


def plot_interactive(threshold):
    return plotly.figure_factory.create_dendrogram(
        iris_data,
        orientation="right",
        labels=agglomerative_clustering.labels_,
        color_threshold=threshold,
    )


# def streamlit_demo():
#     streamlit.header("Cluster Analysis")
#
#     color_threshold = streamlit.slider("Select color threshold.", 0, 15)
#
#     fig = plot_interactive(color_threshold)
#     streamlit.plotly_chart(fig)


def test_dash():
    app = dash.Dash(__name__)

    df = iris

    columns = list(df.columns.values)
    rows = list(df.index)

    app.layout = html.Div(
        [
            "Rows to display",
            dcc.Dropdown(
                id="my-default-clustergram-input",
                options=[{"label": row, "value": row} for row in list(df.index)],
                value=rows[:10],
                multi=True,
            ),
            html.Div(id="my-default-clustergram"),
        ]
    )

    @app.callback(
        Output("my-default-clustergram", "children"),
        Input("my-default-clustergram-input", "value"),
    )
    def update_clustergram(rows):
        if len(rows) < 2:
            return "Please select at least two rows to display."

        return dcc.Graph(
            figure=dashbio.Clustergram(
                data=df.loc[rows].values,
                column_labels=columns,
                row_labels=rows,
                color_threshold={"row": 250, "col": 700},
                hidden_labels="row",
                height=800,
                width=700,
            )
        )


def main():
    # streamlit_demo()

    test_dash()


if __name__ == "__main__":
    main()
