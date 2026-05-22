# instruções

## 1. baixar dependências

```pip install -r requirements.txt```

## 2. inicializar o codigo

### 2.1. windows 

```python main.py```

### 2.2. linux(ubunto)

```python3 main.py```

## 3. explicação

### 3.1 o dataset
Foi utilizado o data set <a>https://archive.ics.uci.edu/dataset/53/iris</a> que contem os dados da seguinte forma:
    <strong>5.1,3.5,1.4,0.2,Iris-setosa</strong>
sendo:
1. Comprimento da sépala em cm
2. Largura da sépala em cm
3. Comprimento da pétala em cm
4. Largura da pétala em cm
5. Classe:
    -- Iris Setosa
    -- Iris Versicolor
    -- Iris Virginica

### 3.1.1 preparar o dataset
Uma ia não entende texto(a classe) e tambem para melhor visualização dos dados eu optei por transformar os dados em uma tabela(via pandas)

```
import pandas as pd

colunas = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]

df = pd.read_csv("iris.data", names=colunas)

df.to_csv("iris_tabela.csv", index=False)
```

