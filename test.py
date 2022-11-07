import numpy as np
import pandas as pd
import preprocess
from sklearn.model_selection import train_test_split


name = "Reference"
data = pd.read_csv(f"data/{name}.csv", delimiter=",")
columns = list(data.columns)
# print(columns)
# print(data.shape[0])
x, y = data.iloc[:, :300].values, data.iloc[:, 300:].values
x, y = preprocess.PreProcess().nonsense_treat(x, y)
print(x.shape[0])
# index = np.random.choice(np.arange(x.shape[0]), 5000, replace=False)
# print(len(set(list(index))))
# x, y = x[index, :], y[index, :]
x_train, x_test, y_train, y_test = train_test_split(x, y,  
                                                    test_size=0.6, random_state=20221103)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
df_train = pd.DataFrame(np.concatenate((x_train, y_train), axis=1), columns=columns)
df_test = pd.DataFrame(np.concatenate((x_test, y_test), axis=1), columns=columns)
df = pd.concat([df_train, df_test], axis=0)
df_train.to_csv(f"data/{name}/{name}-train.csv", header=True, index=False)
df_test.to_csv(f"data/{name}/{name}-test.csv", header=True, index=False)
df.to_csv(f"data/{name}/{name}.csv", header=True, index=False)