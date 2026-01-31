# AGENT_CONTEXT.md - EAI_02 Machine Learning

> **Propósito**: Contexto estruturado para agentes de IA responderem questões sobre Machine Learning Clássico  
> **Última atualização**: Janeiro 2026

## RESUMO EXECUTIVO

**Módulo**: EAI_02 - Machine Learning  
**Objetivo**: Introduzir algoritmos clássicos de ML supervisionado (classificação/regressão)  
**Abordagem**: Progressão de dados sintéticos → reais → projetos completos  
**Nível**: Intermediário (após fundamentos matemáticos)  
**Bibliotecas principais**: scikit-learn, pandas, numpy, matplotlib, seaborn

---

## ESTRUTURA DE ARQUIVOS

```
EAI_02/
├── Fundamentos/
│   ├── analise_exploratoria.ipynb       [EDA, tratamento de dados, pandas]
│   ├── classificacao_sintetica.ipynb    [make_classification, split, scaler]
│   ├── classificacao_KNN.ipynb          [KNN, decision boundaries, k=5]
│   ├── classificacao_projeto_real.ipynb [Breast Cancer, 30 features, 4 modelos]
│   └── comparacao_modelos.ipynb         [LogReg, DT, RF, KNN comparados]
├── Modelo_Base/
│   └── Estrutura_Machine_Learning.ipynb [Template completo, 7 algoritmos]
└── Projetos/
    ├── Desempenho_dos_Alunos/  [Predição acadêmica, student.csv]
    ├── Deteccao_Fraudes/       [Fraudes bancárias, bs140513_032310.csv]
    └── Diabetes/               [Web app, Flask, modelo deployado]
```

---

## NOTEBOOKS - CONTEXTO DETALHADO

### 1. analise_exploratoria.ipynb

**Objetivo**: Ensinar análise exploratória de dados (EDA) e pré-processamento

**Dataset simulado**:
```python
# Estrutura CSV
ID,Nome,Idade,Salario,Departamento
1,Ana,28.0,5200.0,Financeiro
2,Bruno,35.0,NaN,TI
3,Carlos,NaN,4300.0,Marketing
# ... total 5 registros com valores nulos intencionais
```

**Tópicos implementados**:

1. **Informações básicas**:
   - `df.info()` - tipos de dados, memória
   - `df.describe()` - estatísticas descritivas
   - `df.shape` - dimensões (5, 5)

2. **Valores nulos**:
   - `df.isnull().sum()` - contagem de NaN
   - `df.dropna()` - remover linhas
   - `df.fillna(value)` - preencher

3. **Conversão de tipos**:
   - `df['col'].astype(int)` - conversão explícita
   - `pd.to_numeric()` - conversão segura

4. **Valores únicos**:
   - `df['Departamento'].unique()` - categorias
   - `df['Departamento'].value_counts()` - frequência

5. **Preenchimento de faltantes**:
   ```python
   # Média para numéricos
   df['Idade'].fillna(df['Idade'].mean(), inplace=True)
   
   # Moda para categóricos
   df['Departamento'].fillna(df['Departamento'].mode()[0], inplace=True)
   ```

6. **Codificação categórica**:
   ```python
   # Label Encoding
   from sklearn.preprocessing import LabelEncoder
   le = LabelEncoder()
   df['Departamento_encoded'] = le.fit_transform(df['Departamento'])
   # Financeiro=0, Marketing=1, TI=2
   
   # One-Hot Encoding (mencionado)
   pd.get_dummies(df['Departamento'])
   ```

7. **Atualização de valores**:
   ```python
   df.loc[df['Nome'] == 'Ana', 'Salario'] = 6000
   ```

**Bibliotecas usadas**:
- `pandas` - manipulação de dados
- `numpy` - arrays e cálculos
- `io.StringIO` - simular leitura de CSV

**Output**: DataFrame limpo e encodado pronto para ML

---

### 2. classificacao_sintetica.ipynb

**Objetivo**: Introduzir classificação binária com dados controlados

**Dataset**:
```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,      # Número de exemplos
    n_features=2,        # 2 features (x1, x2)
    n_redundant=0,       # Sem features redundantes
    n_informative=2,     # Ambas informativas
    n_clusters_per_class=1,  # 1 cluster por classe
    random_state=42
)
```

**Pipeline implementado**:

1. **Geração de dados**: `make_classification`
2. **Visualização**: 
   ```python
   plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
   plt.savefig("imagens/dados_sinteticos_distribuicao.png")
   ```
3. **Split treino/teste**: 70/30
4. **Normalização**: `StandardScaler`
5. **Modelo**: Regressão Logística (baseline simples)
6. **Avaliação**: Acurácia, matriz de confusão

**Estrutura de dados**:
- X: array (1000, 2) - features
- y: array (1000,) - labels binários {0, 1}

**Bibliotecas**:
- `sklearn.datasets.make_classification`
- `sklearn.model_selection.train_test_split`
- `sklearn.preprocessing.StandardScaler`
- `sklearn.linear_model.LogisticRegression`

---

### 3. classificacao_KNN.ipynb

**Objetivo**: Explorar K-Nearest Neighbors visualmente

**Dataset**: 
- 100 amostras sintéticas
- 2 features (x1, x2)
- 2 classes

**Parâmetros KNN**:
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
```

**Conceitos implementados**:

1. **Geração de dados**: `make_classification(n_samples=100, n_features=2)`

2. **Visualização inicial**:
   ```python
   plt.scatter(X["x1"], X["x2"], c=y, cmap="bwr", edgecolor='k')
   ```

3. **Separação treino/teste**: 70/30
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   ```

4. **Padronização** (CRÍTICO para KNN):
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

5. **Treinamento**:
   ```python
   knn.fit(X_train_scaled, y_train)
   y_pred = knn.predict(X_test_scaled)
   ```

6. **Matriz de confusão**:
   ```python
   from sklearn.metrics import confusion_matrix
   cm = confusion_matrix(y_test, y_pred)
   ```

7. **Visualização de limites de decisão**:
   - Criar meshgrid no espaço 2D
   - Prever classe para cada ponto do grid
   - Plotar contornos com `plt.contourf()`
   - Sobrepor pontos de teste

8. **Predição de novo ponto**:
   ```python
   new_point = np.array([[0.5, 0.5]])
   new_point_scaled = scaler.transform(new_point)
   prediction = knn.predict(new_point_scaled)
   ```

**Considerações sobre KNN**:
- **Lazy learning**: Não há fase de treinamento, apenas memoriza dados
- **Sensível a escala**: Normalização é OBRIGATÓRIA
- **Computacionalmente custoso**: O(n) para predição
- **Parâmetro k**: 
  - k pequeno → overfit (sensível a ruído)
  - k grande → underfit (fronteira muito suave)
  - Regra de ouro: k = √n (mas testar vários valores)

**Bibliotecas**:
- `sklearn.neighbors.KNeighborsClassifier`
- `sklearn.metrics.confusion_matrix`

---

### 4. classificacao_projeto_real.ipynb

**Objetivo**: Aplicar ML em dataset médico real

**Dataset**: Wisconsin Breast Cancer (sklearn)
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

**Características**:
- **Amostras**: 569 tumores
- **Features**: 30 medições numéricas
  - mean radius, mean texture, mean perimeter, mean area
  - mean smoothness, mean compactness, mean concavity
  - mean concave points, mean symmetry, mean fractal dimension
  - + worst/error para cada medição
- **Target**: 
  - 0: Maligno (212 casos)
  - 1: Benigno (357 casos)

**Pipeline completo**:

1. **Carregamento**:
   ```python
   df = pd.DataFrame(data.data, columns=data.feature_names)
   df['target'] = data.target
   ```

2. **Análise exploratória**:
   - `df.head()`, `df.info()`, `df.describe()`
   - Distribuição de classes
   - Correlação entre features

3. **Preparação**:
   ```python
   X = df.drop('target', axis=1)
   y = df['target']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

4. **Modelos treinados**:
   - Regressão Logística
   - Decision Tree
   - Random Forest
   - KNN

5. **Avaliação**:
   ```python
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
   
   accuracy = accuracy_score(y_test, y_pred)
   precision = precision_score(y_test, y_pred)  # Importante: minimizar falsos positivos
   recall = recall_score(y_test, y_pred)        # Importante: detectar todos os malignos
   f1 = f1_score(y_test, y_pred)
   ```

**Interpretação médica**:
- **Precision alta**: Quando digo que é maligno, estou certo
- **Recall alto**: Detecto a maioria dos casos malignos
- **Trade-off**: Em diagnóstico médico, Recall > Precision (melhor falso alarme que erro fatal)

**Bibliotecas**:
- `sklearn.datasets.load_breast_cancer`
- `sklearn.metrics` (classification_report, confusion_matrix)

---

### 5. comparacao_modelos.ipynb

**Objetivo**: Comparar sistematicamente 4 algoritmos

**Dataset**: 
```python
X, y = make_classification(n_samples=1000, n_features=2, random_state=42)
```

**Modelos comparados**:

```python
modelos = {
    "Regressão Logística": LogisticRegression(),
    "Árvore de Decisão": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}
```

**Loop de treinamento e avaliação**:
```python
resultados = {}

for nome, modelo in modelos.items():
    # Treinar
    modelo.fit(X_train_scaled, y_train)
    
    # Prever
    y_pred = modelo.predict(X_test_scaled)
    
    # Avaliar
    resultados[nome] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
```

**Tabela comparativa** (exemplo de output):

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Regressão Logística | 0.93 | 0.92 | 0.94 | 0.93 |
| Árvore de Decisão | 0.89 | 0.88 | 0.90 | 0.89 |
| Random Forest | 0.95 | 0.94 | 0.96 | 0.95 |
| KNN | 0.91 | 0.90 | 0.92 | 0.91 |

**Visualização**:
- 4 matrizes de confusão lado a lado
- Gráfico de barras comparando métricas
- Uso de `seaborn.heatmap()` para matrizes

**Análise típica**:
- **Random Forest**: Melhor performance geral (ensemble)
- **Regressão Logística**: Boa baseline, rápido, interpretável
- **Decision Tree**: Pior que RF (overfit), mas visual
- **KNN**: Bom, mas computacionalmente custoso

**Bibliotecas**:
- `sklearn.linear_model.LogisticRegression`
- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.neighbors.KNeighborsClassifier`
- `seaborn` para visualizações

---

### 6. Estrutura_Machine_Learning.ipynb

**Objetivo**: Template reutilizável para projetos de ML

**Estrutura modular**:

#### Seção 1: Preparação dos Dados
```python
def criar_dados_classificacao():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y

def criar_dados_regressao():
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        noise=0.1,
        random_state=42
    )
    return X, y

def preprocessar_dados(X, y, test_size=0.2, scale=True):
    # Split com stratify para classificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, 
        stratify=y if is_classification else None
    )
    
    # Normalização opcional
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler
```

#### Seção 2: Modelos de Classificação

**Algoritmos implementados**:
1. **Regressão Logística**
   ```python
   LogisticRegression(max_iter=1000)
   ```

2. **Decision Tree**
   ```python
   DecisionTreeClassifier(max_depth=5, random_state=42)
   ```

3. **Random Forest**
   ```python
   RandomForestClassifier(n_estimators=100, random_state=42)
   ```

4. **SVM**
   ```python
   SVC(kernel='rbf', random_state=42)
   ```

5. **Naive Bayes**
   ```python
   GaussianNB()
   ```

6. **KNN**
   ```python
   KNeighborsClassifier(n_neighbors=5)
   ```

7. **Gradient Boosting**
   ```python
   GradientBoostingClassifier(n_estimators=100, random_state=42)
   ```

#### Seção 3: Modelos de Regressão

```python
modelos_regressao = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}
```

#### Seção 4: Avaliação

**Função genérica para classificação**:
```python
def avaliar_classificacao(modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
```

**Função genérica para regressão**:
```python
def avaliar_regressao(modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
```

#### Seção 5: Cross-Validation

```python
def cross_validate_model(modelo, X, y, cv=5):
    scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy')
    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
```

#### Seção 6: Grid Search

```python
def otimizar_hiperparametros(modelo, param_grid, X_train, y_train):
    grid_search = GridSearchCV(
        modelo, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    }

# Exemplo de uso
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}
resultado = otimizar_hiperparametros(
    RandomForestClassifier(random_state=42),
    param_grid,
    X_train, y_train
)
```

#### Seção 7: Clustering (K-Means)

```python
def treinar_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    return {
        'labels': clusters,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_
    }

# Elbow Method
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
```

**Uso do template**:
1. Copiar notebook
2. Substituir funções de criação de dados por carregamento real
3. Ajustar parâmetros dos modelos
4. Executar pipeline completo
5. Selecionar melhor modelo

**Bibliotecas completas**:
```python
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
```

---

## ALGORITMOS - REFERÊNCIA RÁPIDA

### Classificação

**Regressão Logística**
- **Tipo**: Linear
- **Quando usar**: Baseline, problemas lineares, interpretabilidade importante
- **Vantagens**: Rápido, probabilidades calibradas, interpretável
- **Desvantagens**: Assume linearidade
- **Hiperparâmetros**: C (regularização), max_iter

**Decision Tree**
- **Tipo**: Não-linear, baseado em regras
- **Quando usar**: Dados categóricos, interpretabilidade visual
- **Vantagens**: Fácil interpretar, não precisa normalização
- **Desvantagens**: Overfit fácil, instável
- **Hiperparâmetros**: max_depth, min_samples_split, min_samples_leaf

**Random Forest**
- **Tipo**: Ensemble (bagging)
- **Quando usar**: Alta acurácia, dados complexos
- **Vantagens**: Robusto, menos overfit que DT, feature importance
- **Desvantagens**: Black box, lento, muita memória
- **Hiperparâmetros**: n_estimators, max_depth, min_samples_split

**KNN**
- **Tipo**: Instance-based
- **Quando usar**: Dados pequenos, padrões locais
- **Vantagens**: Simples, sem assunções
- **Desvantagens**: Lento para prever, sensível a escala e dimensionalidade
- **Hiperparâmetros**: n_neighbors, metric, weights

**SVM**
- **Tipo**: Margin-based
- **Quando usar**: Alta dimensionalidade, classes bem separadas
- **Vantagens**: Efetivo em alta dimensão, flexível (kernels)
- **Desvantagens**: Lento em grandes datasets, difícil tunear
- **Hiperparâmetros**: C, kernel, gamma

**Naive Bayes**
- **Tipo**: Probabilístico
- **Quando usar**: Texto, features independentes
- **Vantagens**: Rápido, bom com pouco dado
- **Desvantagens**: Assume independência (raramente verdade)
- **Hiperparâmetros**: (poucos, geralmente padrões OK)

**Gradient Boosting**
- **Tipo**: Ensemble (boosting)
- **Quando usar**: Máxima acurácia, competições
- **Vantagens**: Muito acurado, feature importance
- **Desvantagens**: Lento para treinar, overfit se mal configurado
- **Hiperparâmetros**: n_estimators, learning_rate, max_depth

### Regressão

**Linear Regression**
- **Equação**: y = β₀ + β₁x₁ + ... + βₙxₙ
- **Método**: Mínimos quadrados ordinários
- **Quando usar**: Relação linear, baseline

**Decision Tree Regressor**
- **Funcionamento**: Divide espaço em regiões, prediz média
- **Quando usar**: Não-linearidade, interpretabilidade

**Random Forest Regressor**
- **Funcionamento**: Média de múltiplas árvores
- **Quando usar**: Alta acurácia, dados complexos

---

## MÉTRICAS - FÓRMULAS E INTERPRETAÇÃO

### Classificação

**Matriz de Confusão**:
```
                Predito
              |  0  |  1  |
         -----|-----|-----|
Real  0  | TN  | FP  |
      1  | FN  | TP  |
```

**Acurácia**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Interpretação: Proporção de predições corretas

**Precision**:
```
Precision = TP / (TP + FP)
```
Interpretação: De todos que previ como positivo, quantos eram realmente?

**Recall (Sensibilidade)**:
```
Recall = TP / (TP + FN)
```
Interpretação: De todos os positivos reais, quantos detectei?

**F1-Score**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Interpretação: Média harmônica de Precision e Recall

**Quando usar cada métrica**:
- **Accuracy**: Classes balanceadas, custo de erros igual
- **Precision**: Minimizar falsos positivos (spam detection)
- **Recall**: Minimizar falsos negativos (detecção de doenças)
- **F1**: Balancear Precision e Recall

### Regressão

**MSE (Mean Squared Error)**:
```
MSE = (1/n) * Σ(y_true - y_pred)²
```
Interpretação: Penaliza erros grandes quadraticamente

**RMSE (Root Mean Squared Error)**:
```
RMSE = √MSE
```
Interpretação: Mesma unidade da variável target

**R² (Coeficiente de Determinação)**:
```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(y_true - y_pred)²
SS_tot = Σ(y_true - mean(y))²
```
Interpretação: % de variância explicada pelo modelo (0 a 1, maior melhor)

---

## PRÉ-PROCESSAMENTO - CÓDIGO DE REFERÊNCIA

### Split Treino/Teste
```python
from sklearn.model_selection import train_test_split

# Classificação (com stratify para manter proporção de classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% para teste
    random_state=42,    # Reprodutibilidade
    stratify=y          # Manter proporção de classes
)

# Regressão (sem stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)
```

### Normalização
```python
from sklearn.preprocessing import StandardScaler

# Criar scaler
scaler = StandardScaler()

# Fit APENAS no treino
X_train_scaled = scaler.fit_transform(X_train)

# Transform no teste (usando parâmetros do treino)
X_test_scaled = scaler.transform(X_test)

# NUNCA: scaler.fit_transform(X_test) → Data leakage!
```

### Encoding Categórico
```python
# Label Encoding (ordinal)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-Hot Encoding (nominal)
import pandas as pd
df_encoded = pd.get_dummies(df, columns=['categoria'])

# Ou com sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
X_encoded = ohe.fit_transform(X_categorical)
```

### Tratamento de Nulos
```python
# Preencher com média
df['coluna'].fillna(df['coluna'].mean(), inplace=True)

# Preencher com mediana (melhor para outliers)
df['coluna'].fillna(df['coluna'].median(), inplace=True)

# Preencher com moda (categórico)
df['coluna'].fillna(df['coluna'].mode()[0], inplace=True)

# Remover linhas
df.dropna(inplace=True)

# Remover colunas
df.dropna(axis=1, inplace=True)
```

---

## PERGUNTAS FREQUENTES - RESPOSTAS PARA AGENTES

**Q: Quando usar KNN vs Regressão Logística?**
A: KNN para padrões locais complexos (mas caro). Regressão Logística para baseline rápido e interpretável. Se os dados são linearmente separáveis, LogReg é melhor.

**Q: Por que normalizar dados antes do KNN?**
A: KNN usa distância euclidiana. Se uma feature tem escala 0-1000 e outra 0-1, a primeira dominará o cálculo. StandardScaler garante que todas contribuam igualmente.

**Q: Diferença entre Random Forest e Decision Tree?**
A: Decision Tree: 1 árvore, overfit fácil. Random Forest: muitas árvores treinadas em subsets aleatórios, voto majoritário, mais robusto.

**Q: Qual métrica usar: Accuracy ou F1?**
A: Accuracy se classes balanceadas. F1 se desbalanceadas ou se Precision e Recall são igualmente importantes.

**Q: O que é data leakage?**
A: Usar informações do conjunto de teste no treino. Exemplo: `scaler.fit_transform(X_test)` ou incluir dados de teste em qualquer cálculo antes da separação.

**Q: Como evitar overfitting?**
A: (1) Mais dados de treino, (2) Regularização (L1/L2), (3) Validação cruzada, (4) Early stopping, (5) Simplificar modelo (menos features, max_depth menor)

**Q: Quando usar cross-validation?**
A: Sempre que possível para métricas mais confiáveis. Especialmente importante com datasets pequenos (<10k amostras).

**Q: GridSearchCV é sempre melhor?**
A: Não. É exaustivo e lento. Para espaços grandes de hiperparâmetros, usar RandomizedSearchCV ou otimização bayesiana (Optuna).

**Q: Por que Random Forest é "black box"?**
A: Centenas de árvores fazem difícil explicar uma predição específica. Podemos ver feature importance geral, mas não o caminho de decisão como em 1 árvore.

**Q: Qual o melhor algoritmo de ML?**
A: Não existe. Depende do problema. Regra: comece com Regressão Logística (baseline), teste Random Forest (geralmente ganha), otimize o melhor.

---

## PRÓXIMOS MÓDULOS - CONEXÕES

**Para EAI_03 (Deep Learning)**:
- Mesmo pipeline: dados → modelo → avaliação
- Métricas idênticas para classificação
- Random Forest vs Neural Network: trade-off interpretabilidade/acurácia
- Normalização ainda mais crítica em redes neurais

**Para EAI_04 (NLP Clássico)**:
- TF-IDF → vetor numérico → algoritmos deste módulo
- Naive Bayes excelente para classificação de texto
- Baseline: sempre testar LogReg ou RF antes de Transformers

**Para EAI_06 (Visão Computacional)**:
- Features extraídas de imagens → classificadores deste módulo
- Transfer learning: CNN features → Random Forest classifier

**Para EAI_08 (MLOps)**:
- Projetos deste módulo serão deployados
- Pipeline de treinamento → modelo pkl → API Flask
- Monitoramento de métricas em produção

---

## TAGS DE BUSCA

`#machine-learning` `#classificacao` `#regressao` `#scikit-learn` `#knn` `#random-forest` `#decision-tree` `#logistic-regression` `#metricas` `#accuracy` `#precision` `#recall` `#f1-score` `#confusion-matrix` `#cross-validation` `#grid-search` `#normalizacao` `#standard-scaler` `#train-test-split` `#eda` `#pandas` `#data-preprocessing`

---

**Versão**: 1.0  
**Compatibilidade**: Agentes de IA com conhecimento de ML e Python  
**Uso recomendado**: Responder perguntas sobre implementação, conceitos, debugging, ou seleção de algoritmos em ML clássico
