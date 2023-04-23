from jax import random
import jax.numpy as np
import pandas as pd
import matplotlib.pyplot as plt

key = random.PRNGKey(12345)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve

df = pd.DataFrame({
    "var1": np.concatenate([
        random.normal(key, (50,)) * 1.5 + 3.0,
        random.normal(key, (50,)) * 2.0 - 4.0]),
    "var2": random.uniform(key, (100,)),
    "var3": np.concatenate([
        random.normal(key, (50,)) * 2.0 - 2.0,
        random.normal(key, (50,)) * 0.8 + 1.5])
})

score = 4.0 + df["var1"] - df["var3"]
Y = score >= 0

fig1, ax1 = plt.subplots()
ax1.plot(df.loc[Y, "var1"], df.loc[Y, "var3"], "ko", label="True data")
ax1.plot(df.loc[~Y, "var1"], df.loc[~Y, "var3"], "kx", label="False data")
ax1.legend()
ax1.set_xlabel("var1")
ax1.set_ylabel("var3")
ax1.set_title("Scatter plot of var3 against var1")

plt.show()

model = LogisticRegression()
model.fit(df, Y)

test_df = pd.DataFrame({
    "var1": np.concatenate([
        random.normal(key, (25,)) * 1.5 + 3.0,
        random.normal(key, (25,)) * 2.0 - 4.0]),
    "var2": random.uniform(key, (50,)),
    "var3": np.concatenate([
        random.normal(key, (25,)) * 2.0 - 2.0,
        random.normal(key, (25,)) * 0.8 + 1.5])
})

test_scores = 4.0 + test_df["var1"] - test_df["var3"]
test_Y = test_scores >= 0

test_predicts = model.predict(test_df)

plt.show()

print(classification_report(test_Y, test_predicts))
