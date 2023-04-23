from jax import random
import jax.numpy as np

key = random.PRNGKey(12345)

data = np.arange(15)
probabilities = np.array([0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.025,
                          0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025])

assert np.round(np.sum(probabilities), 10) == 1.0, "Probabilities must sum to 1"

selected = random.choice(key, data, p=probabilities, replace=True)
# 0

selected_array = random.choice(key, data, (5, 5), p=probabilities, replace=True)
# array([[ 1, 6, 4, 1, 1],
#       [ 2, 0, 4, 12, 0],
#       [12, 4, 0, 1, 10],
#       [ 4, 1, 5, 0, 0],
#       [ 0, 1, 1, 0, 7]])
