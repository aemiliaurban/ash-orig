import streamlit
from common.data_parser import RDataParser
from common.input_data import INPUT_DATA_DENDROGRAM
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

r = RDataParser(INPUT_DATA_DENDROGRAM)
r.convert_merge_matrix()
r.add_joining_height()
dendrogram(r.merge_matrix)

streamlit.header("Cluster Analysis")

streamlit.pyplot(plt)
