from jax import random
import jax.numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

key = random.PRNGKey(12345)

p_vars = pd.DataFrame({
    "const": np.ones((100,)),
    "X1": random.uniform(key, (100,), minval=0, maxval=15),
    "X2": random.uniform(key, (100,), minval=0, maxval=25),
    "X3": random.uniform(key, (100,), minval=5, maxval=25)
})

residuals = random.normal(key, (100,)) * 12.0
Y = -10.0 + 5.0 * p_vars["X1"] - 2.0*p_vars["X2"] + residuals

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, tight_layout=True)
ax1.scatter(p_vars["X1"], Y, c="k")
ax2.scatter(p_vars["X2"], Y, c="k")
ax3.scatter(p_vars["X3"], Y, c="k")

ax1.set_title("Y against X1")
ax1.set_xlabel("X1")
ax1.set_ylabel("Y")
ax2.set_title("Y against X2")
ax2.set_xlabel("X2")
ax3.set_title("Y against X3")
ax3.set_xlabel("X3")

plt.show()

model = sm.OLS(Y, p_vars).fit()
print(model.summary())

second_model = sm.OLS(Y, p_vars.loc[:, "const":"X2"]).fit()
print(second_model.summary())
