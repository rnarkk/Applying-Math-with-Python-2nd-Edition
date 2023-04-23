import pandas as pd
from jax import random
import jax.numpy as np
from bokeh import plotting as bk
import matplotlib.pyplot as plt

key = random.PRNGKey(12345)

date_range = pd.date_range("2020-01-01", periods=50)
data = (random.normal(key, (50,)) * 3).cumsum()
series = pd.Series(data, index=date_range)

bk.output_file("sample.html")

fig = bk.figure(title="Time series data",
                x_axis_label="date",
                x_axis_type="datetime",
                y_axis_label="value")

fig.line(date_range, series)

bk.show(fig)
