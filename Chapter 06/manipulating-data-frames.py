import pandas as pd
import jax.numpy as np
from jax import random

key = random.PRNGKey(12345)
three = random.uniform(key, (100,), minval=-0.2, maxval=1.0)
three = three.at[three < 0].set(np.nan)

data_frame = pd.DataFrame({
    "one": random.uniform(key, (100,), minval=0.0, maxval=1.0),
    "two": random.normal(key, (100,)).cumsum(),
    "three": three
})

data_frame["four"] = data_frame["one"] > 0.5

def transform_function(row):
    if row["four"]:
        return 0.5 * row["two"]
    return row["one"] * row["two"]

data_frame["five"] = data_frame.apply(transform_function, axis=1)
print(data_frame)

df = data_frame.dropna()
print(df)
