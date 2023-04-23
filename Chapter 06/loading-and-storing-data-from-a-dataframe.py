import pandas as pd
from jax import random
import jax.numpy as np

key = random.PRNGKey(12345)

diffs = random.normal(key, (100,))
cumulative = diffs.cumsum()

data_frame = pd.DataFrame({
    "diffs": diffs,
    "cumulative": cumulative
})
print(data_frame)

data_frame.to_csv("sample.csv", index=False)

df = pd.read_csv("sample.csv", index_col=False)
print(df)
