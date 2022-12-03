import plotly.figure_factory
import streamlit
from hierarchical_clustering import example, iris_example

import dash
from dash.dependencies import Input, Output
import dash_bio as dashbio
from dash import html, dcc

X, labels, Z, dendrogram = example()

iris, iris_data, agglomerative_clustering, iris_dendrogram = iris_example()


app = dash.Dash(__name__)

df = iris
print(df)

columns = list(df.columns.values)
rows = list(df.index)
print(rows)

app.layout = html.Div(
    [
        "Rows to display",
        dcc.Dropdown(
            id="choose_dataset_attributes",
            options=[{"label": column, "value": column} for column in columns],
            value=columns,
            multi=True,
        ),
        html.Div(id="my-default-clustergram"),
    ]
)


@app.callback(
    Output("my-default-clustergram", "children"),
    Input("choose_dataset_attributes", "value"),
    # Input("my-default-clustergram-input", "value")
)
def update_clustergram(columns):
    if len(columns) < 2:
        return "Please select at least two rows to display."

    return dcc.Graph(
        figure=dashbio.Clustergram(
            data=df[columns],
            column_labels=columns,
            row_labels=rows,
            color_threshold={"row": 250, "col": 700},
            hidden_labels="row",
            height=800,
            width=700,
        )
    )


if __name__ == "__main__":
    app.run_server(debug=True)
