# %% import libraries
import torch

from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
# %% generate predictions
y = torch.Tensor([3, 1]).long()
y_pred = torch.Tensor(
    [[0.6891, 0.7101, 0.4097, 0.9859, 0.6466],
    [0.2039, 0.6032, 0.1558, 0.7958, 0.8186]]
)
# %% test macro average accuracy
# accuracy counts as 0 missing classes
acc_obj = Accuracy(task='multiclass', top_k=2, average='macro', num_classes=y_pred.size(1))
print(f'Accuracy: {acc_obj(y_pred, y)}')

def acc(a, b):
    return accuracy(a, b,
                    num_classes=y_pred.size(1),
                    top_k=2,
                    average='macro',
                    task='multiclass')

print(f'functional.accuracy: {acc(y_pred, y)}')
# %%
