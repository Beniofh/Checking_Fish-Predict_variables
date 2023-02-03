# %% import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from data.surface_temperature import load_dataset


# %% loading data
X, y = load_dataset('../data/data_ben.csv', split=False)

# %% fitting model
model = LogisticRegression(max_iter=1000, C=0.1)
poly = PolynomialFeatures(degree=3)
pipe = Pipeline([('Polynomial', poly), ('Logistic', model)])

kf = KFold(n_splits=5, shuffle=True)

train_score = test_score = 0.

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f'**** FOLD-{i+1} ****')
    model.fit(X.iloc[train_index], y[train_index])
    test_score += model.score(X.iloc[test_index], y[test_index])
    print(f'Accuracy logistic reg sur test : {model.score(X.iloc[test_index], y[test_index])}')
    train_score += model.score(X.iloc[train_index], y[train_index])
    print(f'Accuracy logistic reg sur train : {model.score(X.iloc[train_index], y[train_index])}')

print(f'Total accuracy on train : {train_score/5}')
print(f'Total accuracy on test : {test_score/5}')
