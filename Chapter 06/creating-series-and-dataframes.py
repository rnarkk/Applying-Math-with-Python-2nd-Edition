import pandas as pd
from jax import random
import jax.numpy as np

key = random.PRNGKey(12345)

diff_data = random.normal(key, (100,))
cumulative = diff_data.cumsum() 

data_series = pd.Series(diff_data)
print(data_series)

data_frame = pd.DataFrame({
    "diffs": data_series, 
    "cumulative": cumulative 
})

print(data_frame)
