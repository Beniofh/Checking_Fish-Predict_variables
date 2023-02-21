# %% import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score

# %% loading dataframe
df = pd.read_csv('../data/dataset.csv', sep=',', low_memory=False)



# %% selecting feature
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
    'salinity_4.2km_mean_month_band_15_sd'
    ]

qualitative_col = [
    'ecoregion'
]
for col in quantitative_col:
    df = df[df[col] != 'occurence_out_time']
df.reset_index(inplace=True)

# %% processing labels
le = LabelEncoder()
le.fit(df.species.unique())
df['labels'] = le.transform(df.species).astype(int)

# %% Processing NaN on selected features
for col in quantitative_col:
    df.loc[df[col].isna(), col] = df[~df[col].isna()][col].astype(float).mean()

# %% processing qualitive columns
ohe = OneHotEncoder()
ohe.fit(df[['ecoregion']])
X_qual = ohe.transform(df[['ecoregion']])
onehot = pd.DataFrame(X_qual.todense())
onehot.columns = [f'eco_{i}' for i in onehot.columns]
df = pd.concat([df, onehot], axis=1)

# %%
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
n = 7000
k = 10
top10 = top_k_accuracy_score(y_val,
                             rf.predict_proba(X_val),
                             k=k,
                             labels=range(205))
print(f'Top-10 accuracy : {top10}')
# %%
