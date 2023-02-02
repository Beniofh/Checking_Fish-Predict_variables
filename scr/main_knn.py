# %% import libraries
from sklearn.neighbors import KNeighborsClassifier
from data.surface_temperature import load_dataset


# %% loading data
X_train, X_test, y_train, y_test = load_dataset('../data/data_ben.csv')

# %% fitting model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# %% testing model
print(f'Accuracy knn classifier sur test : {knn.score(X_test, y_test)}')
print(f'Accuracy knn classifier sur train : {knn.score(X_train, y_train)}')

# %%
