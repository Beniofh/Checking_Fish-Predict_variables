"""
Train a random forest to obtain a baseline 
top-k Macro and micro average.
The model is a random forest.
"""
# %% import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score

# %% loading dataframe
df = pd.read_csv('../data/dataset.csv', sep=',', low_memory=False)
# %% Feature selection. The features that will be used by the model
quantitative_col = [
    'bathymetry_band_0_mean_9x9',
    'bathymetry_band_0_sd',
    'chlorophyll_concentration_1km_band_0_mean_15x15',
    'chlorophyll_concentration_1km_band_0_sd',
    'salinity_4.2km_mean_year_band_0_mean_7x7',
    'salinity_4.2km_mean_year_band_0_sd',
    'chlorophyll_concentration_1km_band_0_mean_15x15',
    'chlorophyll_concentration_1km_band_0_sd',
    'salinity_4.2km_mean_year_band_0_mean_7x7',
    'salinity_4.2km_mean_year_band_0_sd',
    'salinity_4.2km_mean_month_band_15_mean_3x3',
    'salinity_4.2km_mean_month_band_15_sd',
    'salinity_4.2km_mean_month_band_7_mean_7x7',
    'salinity_4.2km_mean_month_band_7_sd'
    ]

qualitative_col = [
    'ecoregion'
]
for col in quantitative_col:
    df = df[df[col] != 'occurence_out_time']
df.reset_index(inplace=True)

# %% processing labels.
# The labels will be consecutive numbers between 0 and the number of species
le = LabelEncoder()
le.fit(df.species.unique())
df['labels'] = le.transform(df.species).astype(int)

# %% Processing NaN on selected features
# This step can be enhanced. Here NaN are replace by the average
# of the column values that are not NaN

for col in quantitative_col:
    df.loc[df[col].isna(), col] = df[~df[col].isna()][col].astype(float).mean()

# %% processing qualitive columns
# the qualitative variables will be replaced by a vector 
# which size is the number of modality.. All dimension will be set
# at 0 except the one corresponding to the actual current eco region
ohe = OneHotEncoder()
ohe.fit(df[['ecoregion']])
X_qual = ohe.transform(df[['ecoregion']])
onehot = pd.DataFrame(X_qual.todense())
onehot.columns = [f'eco_{i}' for i in onehot.columns]
df = pd.concat([df, onehot], axis=1)

# %% prepare dataset to train model
# the column subset contains values train, val and test
# we will test the model on train and val.
X = df[quantitative_col+onehot.columns.tolist()]
y = df.labels
X_train = X[df.subset=='train']
y_train = y[df.subset=='train']
X_val = X[df.subset=='val']
y_val = y[df.subset=='val']
# %% test random forest
rf = RandomForestClassifier(max_depth=10)
rf.fit(X_train, y_train)
print(f'Top-1 accuracy : {rf.score(X_val, y_val)}')

# %% top-k accuracy
prior = df.groupby('labels')[['id']].count().sort_values('id', ascending=False)
prior.loc[:, 'id'] /= prior['id'].sum()
for k in (1, 5, 10):
    topk = top_k_accuracy_score(y_val,
                                rf.predict_proba(X_val),
                                k=k,
                                labels=range(205))
    print(f'(Micro avg) Top-{k} accuracy : {topk} (prior : {prior.iloc[:k]["id"].sum()})')
# %% Macro average top-K accuracy weights computation
# each species must contribute with the same overall weight in macro average.
# we compute the number of occurence by species. The weight of a species is
# 1/number_of_occurences. This if one species has 10 occurrences each, successfully
# predicted by the model, the accuracy contribution will be 10/10/205=1/205.
dfw = df[df.subset == 'val']
weights = dfw.groupby('labels')\
             .count()[['id']]\
             .apply(lambda a: 1/a).rename({'id': 'weight'}, axis=1)
weights.columns = ['weight']
dfw = dfw.join(weights, how='left', on='labels')

# %% compute weighted top-K
for k in (1, 5, 10):
    topk = top_k_accuracy_score(y_val,
                                rf.predict_proba(X_val),
                                k=k,
                                labels=range(205),
                                sample_weight=dfw.weight)
    print(f'(Macro avg) Top-{k} accuracy : {topk} (prior : {k/df.labels.nunique()})')
# %%
