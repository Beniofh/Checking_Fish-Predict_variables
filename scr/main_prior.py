# %%
import pandas as pd

# %% load data
df = pd.read_csv('../data/data_ben.csv', low_memory=False)
print(f'Nombre d\'espèces : {df.speciesKey.nunique()}')

# %% group by

dfgb = df.groupby('species')[['id']].count()
dfgb = dfgb.sort_values('id', ascending=False)

print(f'Top-1 : {dfgb["id"].iloc[0:1].sum()/dfgb["id"].sum()}')
print(f'Top-5 : {dfgb["id"].iloc[0:5].sum()/dfgb["id"].sum()}')
print(f'Top-10 : {dfgb["id"].iloc[0:10].sum()/dfgb["id"].sum()}')

print(f'{dfgb/dfgb.id.sum()}')