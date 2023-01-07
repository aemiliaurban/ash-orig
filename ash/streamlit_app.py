import plotly
import plotly.figure_factory as ff
import plotly.figure_factory._dendrogram as original_dendrogram
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


def modified_dendrogram_traces(
    self, X, colorscale, distfun, linkagefun, hovertext, color_threshold
):
    """
    Calculates all the elements needed for plotting a dendrogram.

    :param (ndarray) X: Matrix of observations as array of arrays
    :param (list) colorscale: Color scale for dendrogram tree clusters
    :param (function) distfun: Function to compute the pairwise distance
                               from the observations
    :param (function) linkagefun: Function to compute the linkage matrix
                                  from the pairwise distances
    :param (list) hovertext: List of hovertext for constituent traces of dendrogram
    :rtype (tuple): Contains all the traces in the following order:
        (a) trace_list: List of Plotly trace objects for dendrogram tree
        (b) icoord: All X points of the dendrogram tree as array of arrays
            with length 4
        (c) dcoord: All Y points of the dendrogram tree as array of arrays
            with length 4
        (d) ordered_labels: leaf labels in the order they are going to
            appear on the plot
        (e) P['leaves']: left-to-right traversal of the leaves

    """
    from plotly import exceptions, optional_imports

    np = optional_imports.get_module("numpy")
    scp = optional_imports.get_module("scipy")
    sch = optional_imports.get_module("scipy.cluster.hierarchy")
    scs = optional_imports.get_module("scipy.spatial")
    sch = optional_imports.get_module("scipy.cluster.hierarchy")
    # d = distfun(X)
    # Z = linkagefun(d)
    P = sch.dendrogram(
        X,
        orientation=self.orientation,
        labels=self.labels,
        no_plot=True,
        color_threshold=color_threshold,
        truncate_mode="level",
        p=2,
    )

    icoord = scp.array(P["icoord"])
    dcoord = scp.array(P["dcoord"])
    ordered_labels = scp.array(P["ivl"])
    color_list = scp.array(P["color_list"])
    colors = self.get_color_dict(colorscale)

    trace_list = []

    for i in range(len(icoord)):
        # xs and ys are arrays of 4 points that make up the '∩' shapes
        # of the dendrogram tree
        if self.orientation in ["top", "bottom"]:
            xs = icoord[i]
        else:
            xs = dcoord[i]

        if self.orientation in ["top", "bottom"]:
            ys = dcoord[i]
        else:
            ys = icoord[i]
        color_key = color_list[i]
        hovertext_label = None
        if hovertext:
            hovertext_label = hovertext[i]
        trace = dict(
            type="scatter",
            x=np.multiply(self.sign[self.xaxis], xs),
            y=np.multiply(self.sign[self.yaxis], ys),
            mode="lines",
            marker=dict(color=colors[color_key]),
            text=hovertext_label,
            hoverinfo="text",
        )

        try:
            x_index = int(self.xaxis[-1])
        except ValueError:
            x_index = ""

        try:
            y_index = int(self.yaxis[-1])
        except ValueError:
            y_index = ""

        trace["xaxis"] = "x" + x_index
        trace["yaxis"] = "y" + y_index

        trace_list.append(trace)

    return trace_list, icoord, dcoord, ordered_labels, P["leaves"]


original_dendrogram._Dendrogram.create_dendrogram = modified_dendrogram_traces
fig = ff.create_dendrogram(r.merge_matrix)
fig.update_layout(width=800, height=500)
f = fig.full_figure_for_development(warn=False)
fig.show()
