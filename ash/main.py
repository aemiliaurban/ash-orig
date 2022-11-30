import streamlit
import altair

from hierarchical_clustering import example
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

X, labels, Z, dendrogram = example()


def main():
    streamlit.header("Cluster Analysis")
    # streamlit.write(dendrogram)

    streamlit.slider("Select color threshold.", 0, 15)

    plt.figure()

    hierarchy.set_link_color_palette(["m", "c", "y", "k"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    dn1 = hierarchy.dendrogram(
        Z, ax=axes[0], above_threshold_color="y", orientation="top"
    )
    dn2 = hierarchy.dendrogram(
        Z, ax=axes[1], above_threshold_color="#bcbddc", orientation="right"
    )
    hierarchy.set_link_color_palette(None)  # reset to default after use
    plt.show()
    streamlit.pyplot(fig)

    altair_dendrogram = altair.Chart(dendrogram)


if __name__ == "__main__":
    main()
