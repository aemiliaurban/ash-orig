from unittest.mock import patch

import ash.common.data_parser as data_parser
#from ..common import data_parser
from ash.common.input_data import INPUT_DATA_DENDROGRAM
from ash.common.plotly_modified_dendrogram import create_dendrogram_modified
from ash.dash_demo import plot_master

import pandas as pd
import matplotlib.pyplot as plt
import js
import json
import plotly
import plotly.express as px

r = data_parser.RDataParser(INPUT_DATA_DENDROGRAM)
r.convert_merge_matrix()
r.add_joining_height()


def create_dendrogram(value):
    with patch(
        "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
        new=create_dendrogram_modified,
    ) as create_dendrogram:
        fig = plot_master.plot_interactive(create_dendrogram, r.merge_matrix, value)
    return fig.to_json()
