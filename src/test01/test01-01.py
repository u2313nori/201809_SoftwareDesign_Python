# %matplotlib inline

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ボストンのデータセット
from sklearn.datasets import load_boston

boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)

df.insert(0, 'MEDV', boston.target)

df.head()
df.describe()

pd.plotting.scatter_matrix(df, figsize=(15, 15))

df['RM'].hist(bins=20)

# plt.show()

# 終了コメント
print ("終了")
