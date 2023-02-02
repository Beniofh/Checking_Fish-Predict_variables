import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_dataset(path: str, *args, test_size=0.1, split: bool = True, **kwargs) -> pd.DataFrame:
    # loading data
    df = pd.read_csv(path, low_memory=False, *args, **kwargs)

    # encoding labels
    le = LabelEncoder()
    le.fit(df.species.unique())
    df['labels'] = le.transform(df.species).astype(int)

    # cleaning NA
    na = df.mean_15x15_med_sst.isna()
    df.loc[na, 'mean_15x15_med_sst'] = df[~na].mean_15x15_med_sst.mean()

    nac = df.central_value_med_sst.isna()
    df.loc[nac, 'central_value_med_sst'] = df[nac].mean_15x15_med_sst

    namm = df.mean_med_sst.isna()
    df.loc[namm, 'mean_med_sst'] = df[nac].mean_med_sst.mean()

    nastd = df.sd_med_sst.isna()
    df.loc[nastd, 'sd_med_sst'] = df[~nastd].sd_med_sst.mean()

    # X, y
    X = df[['mean_15x15_med_sst', 'central_value_med_sst', 'mean_med_sst', 'sd_med_sst']]
    y = df.labels
    if split:
        #  splitting dataset X_train, X_test, y_train, y_test
        return train_test_split(X, y, test_size=test_size)
    return X, y
