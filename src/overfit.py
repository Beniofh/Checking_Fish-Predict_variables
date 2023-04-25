# %% import libraries
import ast

import numpy as np
import pandas as pd

# import itertools

from sklearn.neighbors import KNeighborsRegressor
# from sklearn.multioutput import MultiOutputRegressor
# %%
df = pd.read_csv('../data/Galaxy117-Sort_on_data_82_n_vec_value.csv')
# %%
ast.literal_eval(df.species_abundance_vec.iloc[0])
# %%
def metric_challenge(y, y_hat, n_dim=True):
    y = y.copy()
    y_hat = y_hat.copy()
    y_hat[y_hat < 0] = 0
    y[y < 0] = 0
    if n_dim:
        S = np.abs(y_hat-y).mean(axis=1)
        return S.mean()
    else:
        return np.abs(y_hat-y)

# %%
df_test = df[df.subset == 'val']
y_test = [ast.literal_eval(r[1]['species_abundance_vec']) for r in df_test.iterrows()]
y_test = np.array(y_test).astype(float)
print(y_test.shape)
df_train = df[df.subset == 'train']
y_train = [ast.literal_eval(r[1]['species_abundance_vec']) for r in df_train.iterrows()]
y_train = np.array(y_train).astype(float)
print(y_train.shape)
# %% testing that the metric gives 0
y_test_log = y_test.copy()
y_test_log[y_test_log == 0] = -float('inf')
y_test_log[y_test_log != -float('inf')] = np.log(y_test_log[y_test_log != -float('inf')])
y_test_log[y_test_log < -1] = -1
print(metric_challenge(y_test_log, y_test_log))


y_train_log = y_train.copy()
y_train_log[y_train_log == 0] = -float('inf')
y_train_log[y_train_log != -float('inf')] = np.log(y_train_log[y_train_log != -float('inf')])
y_train_log[y_train_log < -1] = -1
# %%
feature_names = df_train.columns[11:]
X_train = df_train[feature_names].to_numpy()
X_test = df_test[feature_names].to_numpy()
# %%
mask = np.isnan(X_train).any(axis=0)

scores = []
for sp in range(y_test_log.shape[1]):
    y_sel = None
    best_score = None
    for k in range(1, 15):
        clf = KNeighborsRegressor(n_neighbors=k)
        clf.fit(X_test[:, 1-mask], y_test_log[:, sp])
        y_log_pred = clf.predict(X_test[:, 1-mask])
        s = metric_challenge(y_log_pred, y_test_log[:, sp], n_dim=False)
        print(s.mean())
        if y_sel is None or best_score > s.mean():
            print('best')
            best_score = s.mean()
            y_sel = s
    scores.append(s)
    print('*' * 30)
scores = np.stack(scores, axis=1)
print(scores.mean(axis=0).mean())

# %%
np.unique(X_test[:, 1-mask], axis=0).shape
# %%
