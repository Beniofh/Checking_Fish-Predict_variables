# %% import libraries
import torch

from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
# %% generate predictions

y = torch.randint(0, 5, size=(100,))
y_pred = torch.rand(size=(100, 5))
# %% test macro average accuracy
# accuracy counts as 0 missing classes
acc = Accuracy(task='multiclass', top_k=2,average='macro', num_classes=5)
print(f'Accuracy: {acc(y_pred, y)}')
acc = accuracy(y_pred, y, num_classes=5, top_k=2, average='macro')
print(f'functional.accuracy: {acc}')
# %%
