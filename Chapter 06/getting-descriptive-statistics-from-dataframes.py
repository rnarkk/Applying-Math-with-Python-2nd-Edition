import pandas as pd
from jax import random
import jax.numpy as np
import matplotlib.pyplot as plt

key = random.PRNGKey(12345)

uniform = random.uniform(key, (100,), minval=1, maxval=5)
normal = random.normal(key, (100,)) * 2.5
bimodal = np.concatenate([random.normal(key, (50,)), random.normal(key, (50,)) + 6])

df = pd.DataFrame({
    "uniform": uniform,
    "normal": normal,
    "bimodal": bimodal
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)

df["uniform"].plot(kind="hist", title="Uniform", ax=ax1, color="k", alpha=0.6)
df["normal"].plot(kind="hist", title="Normal", ax=ax2, color="k", alpha=0.6)
df["bimodal"].plot(kind="hist", title="Bimodal", ax=ax3, bins=20, color="k", alpha=0.6)

descriptive = df.describe()
descriptive.loc["kurtosis"] = df.kurtosis()
print(descriptive)

uniform_mean = descriptive.loc["mean", "uniform"]
normal_mean = descriptive.loc["mean", "normal"]
bimodal_mean = descriptive.loc["mean", "bimodal"]

ax1.vlines(uniform_mean, 0, 20, "k")
ax2.vlines(normal_mean, 0, 25, "k")
ax3.vlines(bimodal_mean, 0, 12,"k")

plt.show()




