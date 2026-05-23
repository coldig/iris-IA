# import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# tópico 3.1
dataset = pd.read_csv("iris_tabela.csv")

# tópico 3.2
X = dataset.drop("class", axis=1)
y = dataset["class"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(X)