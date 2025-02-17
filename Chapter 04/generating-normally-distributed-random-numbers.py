from jax import random
import jax.numpy as np
import matplotlib.pyplot as plt

key = random.PRNGKey(12345)

mu = 5.0  # mean value
sigma = 3.0  # standard deviation
rands = random.normal(key, (10000,)) * sigma + mu

fig, ax = plt.subplots()
ax.hist(rands, bins=20, color="k", alpha=0.6)
ax.set_title("Histogram of normally distributed data")
ax.set_xlabel("Value")
ax.set_ylabel("Density")

def normal_dist_curve(x):
    return 10000 * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

x_range = np.linspace(-5, 15)
y = normal_dist_curve(x_range)
ax.plot(x_range, y, "k--")

plt.show()
