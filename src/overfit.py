# %% import libraries
import ast

import numpy as np
import pandas as pd

# import itertools

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.multioutput import MultiOutputRegressor


# %%
df_ab = pd.read_csv('../data/Galaxy117-Sort_on_data_82_n_vec_all.csv')
df_features = pd.read_csv('../data/Galaxy117-Sort_on_data_82_n_vec_value.csv') \
                .drop('species_abundance_vec', axis=1)

df = df_ab[['SurveyID', 'species_abundance_vec']].set_index('SurveyID') \
                                                 .merge(df_features, how='left', on='SurveyID')

# %% constructing abondance vector

df_test = df[df.subset == 'val']
y_test = [ast.literal_eval(r[1]['species_abundance_vec']) for r in df_test.iterrows()]
y_test = np.array(y_test).astype(float)
print(y_test.shape)
df_train = df[df.subset == 'train']
y_train = [ast.literal_eval(r[1]['species_abundance_vec']) for r in df_train.iterrows()]
y_train = np.array(y_train).astype(float)
print(y_train.shape)

# %% consider "log abondance"
def to_log(abondance):
    abondance = abondance.copy()
    abondance[abondance == 0] = -float('inf')
    abondance[abondance != -float('inf')] = np.log(abondance[abondance != -float('inf')])
    abondance[abondance < -1] = -1
    return abondance

y_test_log = to_log(y_test)
y_train_log = to_log(y_train)

# %% construct_X
feature_names = list(df_train.columns[11:]) + ['SiteLat', 'SiteLong']
X_train = df_train[feature_names].to_numpy()
X_test = df_test[feature_names].to_numpy()

# %% not null
mask = np.isnan(X_train).any(axis=0)
sel_mask = (1-mask).astype(bool)
print(f'Available variables: {(sel_mask).sum()}')
print(f'Not available variables: {mask.sum()}')

# %%
list(df[feature_names].loc[:, sel_mask].columns)

# %%
def metric_challenge(y, y_hat, reduce=True):
    """
    This metric accepts log values...
    """
    y = y.copy()
    y_hat = y_hat.copy()
    y_hat[y_hat < 0] = 0
    y[y < 0] = 0
    if reduce:
        S = np.abs(y_hat-y).mean(axis=1)
        return S.mean()
    else:
        return np.abs(y_hat-y)


# %%


scores = []
for sp in range(y_test_log.shape[1]):
    y_sel = None
    best_score = None
    for k in range(1, 15):
        clf = KNeighborsRegressor(n_neighbors=k)
        clf.fit(X_train[:, sel_mask], y_train_log[:, sp])
        y_log_pred = clf.predict(X_test[:, sel_mask])
        s = metric_challenge(y_log_pred, y_test_log[:, sp], reduce=False)
        print(f'Current score : {s.mean()}')
        if y_sel is None or best_score > s.mean():
            print('New best score')
            best_score = s.mean()
            y_sel = s
    scores.append(y_sel)
    print('*' * 30)
scores = np.stack(scores, axis=1)
print(f'Total score: {scores.mean(axis=1).mean()}')

# %%
print(f'Number of sites: {np.unique(X_test[:, sel_mask], axis=0).shape[0]}')
print(f'Original dimensions: {X_test.shape}')
# %%
clf = RandomForestRegressor(max_depth=10, random_state=0, criterion="squared_error") 
clf.fit(X_test[:, sel_mask], y_test_log)
y_pred = clf.predict(X_test[:, sel_mask])
s = metric_challenge(y_pred, y_test_log)
print(f'Random forest score test on test: {s}')

# %% select one site at two moments
df_temp = df[df.SurveyID.isin([912354162, 912354174])]
# %%
clf = RandomForestRegressor(max_depth=2, random_state=0, criterion="squared_error") 
clf.fit(X_train[:, sel_mask], y_train_log)
y_pred = clf.predict(X_test[:, sel_mask])
s = metric_challenge(y_pred, y_test_log)
print(f'Random forest score train on test: {s}')
# %%
