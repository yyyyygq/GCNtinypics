#%%
import os
os.chdir("C:\\Users\\bigbi\\Desktop\\thesis")

import pandas as pd
train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')


#检查缺失值
print(train.isna().sum().sum(),test.isna().sum().sum())
# %%
