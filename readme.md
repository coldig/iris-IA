# instruções

## 1. Baixar dependências

```pip install -r requirements.txt```

## 2. Inicializar o codigo

### 2.1. windows 

```python main.py```

### 2.2. linux(ubunto)

```python3 main.py```

## 3. Explicação

### 3.1. O dataset
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

### 3.1.1. Preparar o dataset
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

### 3.2. Separar dados de treino e de teste
Então inicialmente retiramos as classes(```dataset["class"]```) e separamos elas em uma variavel Y e o restante dos dados(que serão usados para o treino) em uma variavel X:

```
encoder = LabelEncoder()

X = dataset.drop("class", axis=1)
y = encoder.fit_transform(dataset["class"])
```

_o ```LabelEncoder``` é para transformar os nomes(str) em valores(int)_
Ex:
```
Iris-setosa -> 0
Iris-versicolor -> 1
Iris-virginica -> 2
```

Eu optei por separar 25% dos dados para teste e 75% para treino, e separando com a função ```train_test_split``` a distribuição fica aleatória(e não alguma estatística):
```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

_```random_state``` : como uma "seed" para "controlar" a aleatoriedade_
_```stratify```: preserva a distribuição para evitar <a href="https://www.simplypsychology.org/sampling-bias-types-examples-how-to-avoid-it.html">Sampling bias</a>_

### 3.3. Normalização
Após carregar os dados, e separar devidamente para treino e para testes, agora vem a etapa de normalizar os dados.
A normalização de dados afeta diretamente como o modelo aprende e sem ela o treino fica instável.

A normalização que optei foi a Standardization. Primeiramente instanciamos a classe e chamamos o metodo ```fit_transform```:

```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

_o fit_transform justa as funções fit + transform_
#### fit:
Ele aprende os dados e calcula:
- média de cada coluna
- desvio padrão de cada coluna

#### transform:
Depois de aprender os valores, ele transforma os dados:
```novo_valor = (valor - media) / desvio```
<br>
Então ele é um atalho para:
```
scaler.fit(X)
X = scaler.transform(X)
```

### 3.4. Criando o modelo 
```
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

A variavel model é uma instacia da classe Sequential que como o nome sugere, é um modelo em sequencia, ou seja, define um fluxo encadeado de dados entre camadas, onde a saída de uma camada é a entrada da próxima.

![alt text](https://media.geeksforgeeks.org/wp-content/uploads/20250929154234052438/backpropagation_in_neural_network_8.webp)
_imagem do funcionamento de uma MLP(modelo sequencial)_

#### Dense
```
tf.keras.layers.Dense(16, activation='relu', input_shape=(4,))
```

Dense é uma camada de neurônios(no nosso caso, com 16 neurônios)

<br>

_e o ```activation='relu'```?_
"As funções de ativação são um bloco de construção integral das redes neurais. Eles transformam o sinal de entrada de um nó em uma rede neural em um sinal de saída que é então passado para a próxima camada. Sem as funções de ativação, as redes neurais ficariam restritas à modelagem apenas de relações lineares entre entradas e saídas, por exemplo, por meio da multiplicação de matrizes.

No entanto, a maioria dos dados do mundo real não pode ser modelada com linearidades. As não linearidades capturam padrões como, por exemplo, o fato de que passar de nenhum filho para um filho pode afetar suas transações bancárias de forma diferente de passar de três para quatro filhos. Se as redes neurais não tivessem funções de ativação, elas não conseguiriam aprender os complexos padrões não lineares que existem nos eventos do mundo real."

fonte: https://www.datacamp.com/pt/blog/rectified-linear-unit-relu

<br>

cada neurônio faz uma soma ponderada
$$
z=w1​x1​+w2​x2​+w3​x3​+w4​x4​+b
$$

e com base nos resultados o ReLU faz:
“se isso não for relevante, ignora (zera)”
“se for relevante, deixa passar”

<br>

Exemplo mental:
&nbsp; neurônio 1: detecta “pétala grande”
&nbsp; neurônio 2: detecta “sepal largo”
&nbsp; neurônio 3: detecta “combinação X”

Se não detecta nada relevante:
&nbsp;ReLU zera → neurônio “desligado”

<br>

_```input_shape=(4,)```: a camada espera 4 valores(sepal_length, sepal_width, petal_length, petal_width)_

Cada neurônio pega esses 4 valores, faz soma ponderada, passa pelo ReLU e solta para a próxima camada

#### e porque a segunda camada não tem o ```input_shape```?
```
tf.keras.layers.Dense(16, activation='relu')
```
A segunda camada não precisa de input_shape porque ela já recebe automaticamente a saída da camada anterior, que no caso tem 16 valores.
$$
4 → 16 → 16 → 3
$$

<br>

#### saída ou tomada de decisão
```
tf.keras.layers.Dense(3, activation='softmax')
```
Essa é a camada de output, que transforma tudo aprendido em probabilidades. Possui 3 neurônios pois devolve as chances de serem uma das 3 classes(Iris-setosa, Iris-versicolor, Iris-virginica)
ex:
```[0.10, 0.70, 0.20]```

Ela recebe os resultados dos 16 neurônios e aplica o softmax os transformando em probabilidades

#### oque é o 'softmax'?
Ele é o responsavel por transformar valores $$z=w1​x1​+w2​x2​+...+w16​x16​+b$$ em prababilidades $$[0.66, 0.24, 0.10]$$ 
_formula:_

<img src="https://imgs.search.brave.com/Xssg5I0jTZuFQvSELd9tEiLZTRMxT2d-rnoi6KvmhLM/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YTIuZGV2LnRvL2R5/bmFtaWMvaW1hZ2Uv/d2lkdGg9ODAwLGhl/aWdodD0sZml0PXNj/YWxlLWRvd24sZ3Jh/dml0eT1hdXRvLGZv/cm1hdD1hdXRvL2h0/dHBzOi8vZGV2LXRv/LXVwbG9hZHMuczMu/YW1hem9uYXdzLmNv/bS91cGxvYWRzL2Fy/dGljbGVzL2VxNWJy/cWl5MHdueWVlbnRy/OXFpLnBuZw" width="400" alt="all_text">

### 3.5. Compilação
```
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

É nessa parte que você vai dizer pro modelo:
- "como aprender"
- "como medir erro"
- "como avaliar desempenho"

•```optimizer='adam'```

```optimizer```: é o algoritmo responsavel por ajustar os pesos(fazer o modelo aprender). Sem isso o modelo não aprende.
```adam```: é um dos otimizadores mais usados. Ele basicamente faz a pergunta "qual direção reduz o erro mais rápido?" e ajusta os pesos de forma inteligente.

_O aprendizado da rede neural aprende via backpropagation + optimizer_

•```loss='sparse_categorical_crossentropy'```

```loss```: é a função de erro, ela basicamente mede "o quão errada a IA está". 

```sparse```: define como as labels estão representadas. 

existem duas formas de represetar classes:

##### inteiros normais:
```
0
1
2
```
Ex:
```
setosa -> 0
versicolor -> 1
virginica -> 2
```

##### One-hot encoding:
```
[1,0,0]
[0,1,0]
[0,0,1]
```
Ex:
```
setosa      -> [1,0,0]
versicolor  -> [0,1,0]
virginica   -> [0,0,1]
```

```categorical```: indica que essa loss é para problemas de multiplas categorias(ou classes). Aqui temos 3 classes:
- setosa
- versicolor
- virginica

```crossentropy``` essa loss compara a previsão, vê se a resposta esta correta e gera um numero de erro. é uma formula matematica(Cross Entropy).

•```metrics=['accuracy']```
Define as metricas utilizadas para acompanhar o desempenho do modelo.
exemplo de metricas:
- Accuracy
- Precision
- Recall

e muitas outras.

### 3.6. Treinamento

```
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)
```

```fit```: É responsavel por inicar o treinamento do modelo. 
ele faz um ciclo repetitivo:
1. recebe dados
2. faz previsões
3. calcula o erro (loss)
4. usa backpropagation
5. atualiza os pesos (de cada neurônio)
6. repete isso várias vezes

```X_train```: o primeiro parâmetro da função são as entradas.
```y_train```: ja o segundo são as saídas.

internamente a rede pega o ```X_train```, faz previsões e compara com o ```y_train``` usando a loss.

```epochs```: são quantas vezes a rede repetira o ciclo de treinamento.

_colocar ```epochs``` muito altas podem causar o problema de <a href="https://www.ibm.com/br-pt/think/topics/overfitting">overfitting</a>, que é quando a rede memoria os dados envez de aprender os padrões._