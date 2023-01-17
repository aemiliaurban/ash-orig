import altair
import matplotlib.pyplot as plt
import networkx as nx
import plotly.figure_factory
import streamlit_app
from hierarchical_clustering import example, iris_example
from scipy.cluster import hierarchy
from sklearn import datasets

X, labels, Z, dendrogram = example()

iris, iris_data, iris_labels, iris_dendrogram = iris_example()


def plot_interactive(threshold):
    return plotly.figure_factory.create_dendrogram(
        iris.data, orientation="right", labels=iris_labels, color_threshold=threshold
    )


def main():
    streamlit.header("Cluster Analysis")
    # streamlit.write(dendrogram)

    color_threshold = streamlit.slider("Select color threshold.", 0, 15)
    #
    # plt.figure()
    #
    # # hierarchy.set_link_color_palette(["m", "c", "y", "k"])
    # # fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    # # dn1 = hierarchy.dendrogram(
    # #     Z, ax=axes[0], above_threshold_color="y", orientation="top"
    # # )
    # # dn2 = hierarchy.dendrogram(
    # #     Z, ax=axes[1], above_threshold_color="#bcbddc", orientation="right"
    # # )
    # # hierarchy.set_link_color_palette(None)  # reset to default after use
    # # plt.show()
    # # streamlit.pyplot(fig)
    #
    # altair_dendrogram = altair.Chart(dendrogram)
    # fig, ax = plt.subplots()
    # d = hierarchy.dendrogram(hierarchy.linkage(X, method="ward"))
    # #print(d)
    # #print(type(d))
    # g = nx.DiGraph()
    # g.add_nodes_from(d.keys())
    # pos = nx.kamada_kawai_layout(g)
    # #G = nx.Graph(hierarchy.dendrogram(hierarchy.linkage(X, method="ward")))
    # nx.draw(g, pos)
    # plt.draw()
    # plt.show()
    # print(type(fig))
    fig = plot_interactive(color_threshold)
    streamlit.plotly_chart(fig)

    # streamlit.altair_chart(altair_dendrogram)


#
# options = ["euclidean", "minkowski", "cityblock", "seuclidean", "sqeuclidean",
#            "cosine", "correlation", "hamming", "jaccard", "jensenshannon", "chebyshev", "canberra",
#            "braycurtis", "mahalanobis", "yule", "matching", "dice", "kulczynski1", "rogerstanimoto",
#            "russellrao", "sokalmichener", "sokalsneath"],

if __name__ == "__main__":
    main()
