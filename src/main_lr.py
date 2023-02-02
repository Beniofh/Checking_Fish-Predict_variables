# %% import libraries
from sklearn.linear_model import LogisticRegression
from data.surface_temperature import load_dataset


# %% loading data
X_train, X_test, y_train, y_test = load_dataset('../data/data_ben.csv')

# %% fitting model
model = LogisticRegression(max_iter=1000, C=0.1)
model.fit(X_train, y_train)

# %% testing model
print(f'Accuracy logistic classifier sur test : {model.score(X_test, y_test)}')
print(f'Accuracy logistic classifier sur train : {model.score(X_train, y_train)}')

# %%
