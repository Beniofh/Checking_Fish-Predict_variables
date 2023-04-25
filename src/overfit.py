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

df['ts'] = pd.to_datetime(df.SurveyDate).values.astype(np.int64)// 10**9

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
    abondance[abondance != -float('inf')] = np.log10(abondance[abondance != -float('inf')])
    abondance[abondance < -1] = -1
    return abondance

y_test_log = to_log(y_test)
y_train_log = to_log(y_train)

# %% construct_X
feature_names = list(df_train.columns[11:]) + ['SiteLat', 'SiteLong', 'ts']
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

def fine_tune_on_test(base, param_combination, X_tr, y_tr, X_te, y_te):
    scores = []
    best_params = {}
    for sp in range(y_test.shape[1]):
        y_sel = None
        best_score = None
        best_params[sp] = {}
        for params in param_combination:
            clf = base(**params)
            clf.fit(X_tr[:, sel_mask], y_tr[:, sp])
            y_log_pred = clf.predict(X_te[:, sel_mask])
            s = metric_challenge(y_log_pred, y_te[:, sp], reduce=False)

            if y_sel is None or best_score > s.mean():

                best_score = s.mean()
                best_params[sp] = params
                y_sel = s
        scores.append(y_sel)
    scores = np.stack(scores, axis=1)
    print(f'Total score: {scores.mean(axis=1).mean()}')
    return best_params

# %%
print(f'Number of sites: {np.unique(X_test[:, sel_mask], axis=0).shape[0]}')
print(f'Original dimensions: {X_test.shape}')

# %% testing  KNN Regressor
print('\nKNN\n')
print('*' * 10, 'Test on test', '*' * 10)
fine_tune_on_test(KNeighborsRegressor,
                  [{'n_neighbors': k} for k in range(1, 15)],
                  X_test,
                  y_test_log,
                  X_test,
                  y_test_log)

print('*' * 10, 'Train on test', '*' * 10)
fine_tune_on_test(KNeighborsRegressor,
                  [{'n_neighbors': k} for k in range(1, 15)],
                  X_train,
                  y_train_log,
                  X_test,
                  y_test_log)

# %%
print('\nRandom Forest\n')
p = [{
    'max_depth': i,
    'random_state': 0,
    'criterion': 'squared_error'
} for i in range(8, 11, 2)]

print('*' * 10, 'Test on test', '*' * 10)

fine_tune_on_test(RandomForestRegressor,
                  p,
                  X_test,
                  y_test_log,
                  X_test,
                  y_test_log)

# print('*' * 10, 'Train on test', '*' * 10)

# bp = fine_tune_on_test(RandomForestRegressor,
#                        p,
#                        X_train,
#                        y_train_log,
#                        X_test,
#                        y_test_log)


# %% select one site at two moments
df_temp = df[df.SurveyID.isin([912354162, 912354174])]
# %%
