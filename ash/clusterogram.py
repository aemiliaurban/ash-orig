import timeit
import dash
import dash_bio as dashbio
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from scipy.cluster import hierarchy
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering


def iris_example():
    iris = datasets.load_iris(as_frame=True)
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

    dendrogram = hierarchy.dendrogram(hierarchy.linkage(iris.data, method="ward"))

    model = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
    result = model.fit(iris.data)

    return iris_df, iris.data, result, dendrogram


def example():
    dataset = pd.read_csv("/Users/niki/diplomka/ash/ash/Mall_Customers.csv")

    X = dataset.iloc[:, [3, 4]].values
    Z = hierarchy.linkage(X, "single")
    print(Z)

    dendrogram = hierarchy.dendrogram(hierarchy.linkage(X, method="ward"))

    model = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
    model.fit(X)
    labels = model.labels_

    # plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='red')
    # plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='blue')
    # plt.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='green')
    # plt.scatter(X[labels==3, 0], X[labels==3, 1], s=50, marker='o', color='purple')
    # plt.scatter(X[labels==4, 0], X[labels==4, 1], s=50, marker='o', color='orange')
    # plt.show()

    return X, labels, Z, dendrogram


example()


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
