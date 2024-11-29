#%%
import os
os.chdir("C:\\Users\\bigbi\\Desktop\\thesis")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import torchvision
import numpy as np 
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch_geometric.utils as utils
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

m1 = train.iloc[5,1:785]
m1 = m1.values.flatten()
m1 =  m1.reshape((28,28))
plt.imshow(m1, cmap='hot', interpolation='nearest')
plt.colorbar()  # Show color scale
plt.title('Heat Map of 28x28 Matrix')
plt.show()


mnist_train = pd.read_csv('fashion-mnist_train.csv')
mnist_test = pd.read_csv('fashion-mnist_test.csv')
X_train = mnist_train.iloc[:, 1:].values  # Pixel values
y_train = mnist_train.iloc[:, 0].values   # Labels

X_test = mnist_test.iloc[:, 1:].values  # Pixel values
y_test = mnist_test.iloc[:, 0].values   # Labels

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
# Define the CNN model
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv + ReLU + Pooling
        x = x.view(-1, 16 * 3 * 3)           # Flatten
        x = self.fc1(x)                      # Fully connected layer
        return F.log_softmax(x, dim=1)

# Downsample the dataset to 7x7
def preprocess_mnist_like_data(X, y, size=7):
    # Reshape and normalize
    X = X.reshape(-1, 1, 28, 28).astype('float32') / 255.0  # (N, 1, 28, 28)
    X = torch.tensor(X)
    # Resize to 7x7
    transform = transforms.Resize((size, size))
    X_resized = torch.stack([transform(img) for img in X])
    y = torch.tensor(y, dtype=torch.long)
    return X_resized, y

# Replace with actual MNIST-like data
# X_train, y_train, X_test, y_test are numpy arrays
X_train, y_train = preprocess_mnist_like_data(X_train, y_train, size=7)
X_test, y_test = preprocess_mnist_like_data(X_test, y_test, size=7)

# Prepare datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss, and optimizer
num_classes = len(set(y_train.numpy()))  # Determine number of classes
model = SmallCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import time
# Start the timer
start_time = time.time()
# Training loop
lossdata = [0 for _ in range(10)]

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Calculate and print epoch time
    epoch_time = time.time() - epoch_start_time
    loss = running_loss / len(train_loader)
    lossdata[epoch] = float(loss)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Time: {epoch_time:.2f}s")

# Total training time
total_time = time.time() - start_time
print(f"Total Training Time: {total_time:.2f}s")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = {
    'epoch': list(range(1, 11)),  # 1 到 10 的 epoch
    'loss': lossdata
}
lossdata = pd.DataFrame(lossdata)
# 设置 Seaborn 样式
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

# 绘制折线图
plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x='epoch', y='loss', marker='o', label='Training Loss')

# 添加标题和标签
plt.title('Training Loss Across Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(df['epoch'])  # 设置 x 轴刻度
plt.legend(fontsize=12)
plt.tight_layout()

# 显示图表
plt.show()
#%%
