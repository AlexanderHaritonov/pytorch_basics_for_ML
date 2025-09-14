import pandas as pd
df = pd.read_csv("./league_of_legends_data_large.csv")
df.shape
df.head()
X = df.drop('win', axis=1)
y = df['win']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
display(f'X_train shape: {X_train.shape}')
display(f'y_train shape: {y_train.shape}')
display(f'X_test shape: {X_test.shape}')
display(f'y_test shape: {y_test.shape}')

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long).view(-1, 1).float()
y_test = torch.tensor(y_test.values, dtype=torch.long).view(-1, 1).float()
print('tensors shape:')
display(f'X_train shape: {X_train.shape}, Dtype: {X_train.dtype}')
display(f'y_train shape: {y_train.shape}, Dtype: {y_train.dtype}')
display(f'X_test shape: {X_test.shape}, Dtype: {X_test.dtype}')
display(f'y_test shape: {y_test.shape}, Dtype: {y_test.dtype}')