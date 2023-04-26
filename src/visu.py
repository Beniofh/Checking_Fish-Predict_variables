# %% import libraries
import ast

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# %% loading data

df_ab = pd.read_csv('../data/Galaxy117-Sort_on_data_82_n_vec_all.csv')
df_features = pd.read_csv('../data/Galaxy117-Sort_on_data_82_n_vec_value.csv') \
                .drop('species_abundance_vec', axis=1)

df = df_ab[['SurveyID', 'species_abundance_vec']].set_index('SurveyID') \
                                                 .merge(df_features, how='left', on='SurveyID')

df['ts'] = pd.to_datetime(df.SurveyDate).values.astype(np.int64)// 10**9
# %%
abondance = [ast.literal_eval(r[1]['species_abundance_vec']) for r in df.iterrows()]
abondance = np.array(abondance).astype(float)
df_abondance = pd.DataFrame(abondance)
df_abondance = df_abondance.add_prefix('sp_')
df = pd.concat([df, df_abondance], axis=1)\
       .drop('species_abundance_vec', axis=1)
# %%
# list(df['ts'])
for i in range(99):
    print(f"{i} : {df[f'sp_{i}'].sum()}")

limit = 1596117600

# %%
sites = list(df[df.ts>=1596117600]['SiteCode'].unique())

# %%

df_subset = df[df.SiteCode.isin(sites)]

# %%

sns.set_theme()

species='sp_93'

plt.figure()
sns.lineplot(data=df_subset[(df_subset[species]>0)], x='ts',
             y=species,
             hue='SiteCode',
             errorbar=('ci', 0))
plt.yscale('log')
plt.show()
# %%
