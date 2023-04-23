from jax import random
import jax.numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

key = random.PRNGKey(12345)

SHAPE = (100,)
x_data = random.uniform(key, shape=SHAPE, minval=-3.0, maxval=3.0)
noise = random.normal(key, shape=SHAPE) * 0.8

y_data = 2.0 * x_data ** 2 - 4 * x_data + noise

fig, ax = plt.subplots()
ax.scatter(x_data, y_data, marker="x", color="k", alpha=0.5)
ax.set(xlabel="x", ylabel="y", title="Scatter plot of sample data")

def func(x, a, b, c):
    return a * x ** 2 + b * x + c

coeffs, _ = curve_fit(func, x_data, y_data)
print(coeffs)
# [ 1.99611157 -3.97522213  0.04546998]

x = np.linspace(-3.0, 3.0, SHAPE[0])
y = func(x, coeffs[0], coeffs[1], coeffs[2])
ax.plot(x, y, "k")

plt.show()
