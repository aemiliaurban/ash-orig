import matplotlib.pyplot as plt
import numpy
import plotly.express as px
import plotly.graph_objects as go
from dash import dash, dcc, html
from scipy.cluster.hierarchy import dendrogram, linkage

from ash.common import input_data
from ash.common.data_parser import RDataParser

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
