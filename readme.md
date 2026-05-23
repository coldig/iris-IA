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

e depois é só carregar com:

```df = pd.read_csv("iris_tabela.csv")```

### 3.2. Normalização
Após carregar os dados, agora vem a etapa de normalizar os dados.
A normalização de dados afeta diretamente como o modelo aprende e sem ela o treino fica instável.

então inicialmente retiramos a classe e separamos ela em uma variavel Y e o restante dos dados(que serão usados para o treino) em uma variavel X:

```
X = dataset.drop("class", axis=1)
y = dataset["class"]
```

A normalização que optei foi a Standardization. Primeiramente instanciamos a classe e chamamos a o metodo ```fit_transform```:

```
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

_o fit_transform justa as funções fit + transform_
#### fit:
Ele aprende os dados e calcula:
    <li>média de cada coluna
    <li>desvio padrão de cada coluna

#### transform:
Depois de aprender os valores, ele transforma os dados:
```novo_valor = (valor - media) / desvio```
<br>
Então ele é um atalho para:
```
scaler.fit(X)
X = scaler.transform(X)
```