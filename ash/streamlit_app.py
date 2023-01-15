import plotly
import plotly.figure_factory as ff
import streamlit
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from ash.common.data_parser import RDataParser
from ash.common.input_data import INPUT_DATA


def plot_interactive(threshold, data):
    return plotly.figure_factory.create_dendrogram(
        data,
        orientation="right",
        color_threshold=threshold,
    )


r = RDataParser(INPUT_DATA)
r.convert_merge_matrix()
r.add_joining_height()
dendrogram(r.merge_matrix)

streamlit.header("Cluster Analysis")

color_threshold = streamlit.slider("Select color threshold.", 0, 15)

plt.show()

streamlit.pyplot(plt)

fig = ff.create_dendrogram(r.merge_matrix)
fig.update_layout(width=800, height=500)
f = fig.full_figure_for_development(warn=False)
fig.show()

streamlit.plotly_chart(fig)

