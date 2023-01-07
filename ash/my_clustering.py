import matplotlib.pyplot as plt
import numpy
import plotly.express as px
import plotly.graph_objects as go
from dash import dash, dcc, html
from scipy.cluster.hierarchy import dendrogram, linkage


class RDataParser:
    def __init__(self, input_data):
        self.merge_matrix = [map(float, x) for x in input_data["merge_matrix"]]
        self.joining_height = [float(x) for x in input_data["joining_height"]]
        self.order = [float(x) for x in input_data["order"]]
        self.labels = input_data["labels"]

    def convert_merge_matrix(self):
        transformed_matrix = []
        for node in self.merge_matrix:
            new_node = []
            for el in node:
                if el < 0:
                    transformed_el = abs(el) - 1
                else:
                    transformed_el = el + len(self.merge_matrix)
                new_node.append(transformed_el)
            transformed_matrix.append(new_node)

        self.merge_matrix = transformed_matrix
        print(transformed_matrix)

    def add_joining_height(self):
        # TODO: error for len merge matrix != len joinig height
        for index in range(len(self.merge_matrix)):
            self.merge_matrix[index].append(self.joining_height[index])
            self.merge_matrix[index].append(self.order[index])
        print(self.merge_matrix)


input_data = {
    "merge_matrix": [
        [-1, -8],
        [-3, -5],
        [-6, -10],
        [-4, 3],
        [1, 4],
        [-2, 2],
        [-9, 6],
        [5, 7],
        [-7, 8],
    ],
    "joining_height": [
        282.45,
        537.97,
        629.66,
        844.635,
        1347.43916666667,
        1968.0425,
        3422.88555555556,
        6094.602725,
        19984.8865432099,
    ],
    "order": [7, 1, 8, 4, 6, 1, 0, 9, 2, 3, 5],
    "labels": [
        "Alabama",
        "Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
    ],
}

app = dash.Dash(__name__)

# fig = plt.figure()

r = RDataParser(input_data)
r.convert_merge_matrix()
r.add_joining_height()
dendrogram(r.merge_matrix)

# fig = go.Figure(data=dendrogram(r.merge_matrix))

# app.layout = html.Div([
#     dcc.Graph(figure=fig)
# ])
# app.run_server(debug=True)
