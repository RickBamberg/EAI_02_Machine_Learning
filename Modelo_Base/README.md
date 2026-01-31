# Modelo Base - Machine Learning Clássico

## 📌 Sobre

Este diretório contém **templates reutilizáveis** e **estruturas base** para projetos de Machine Learning. O objetivo é fornecer um ponto de partida organizado e bem documentado para novos projetos.

## 🎯 Objetivo

Facilitar o desenvolvimento de novos projetos de ML fornecendo:
- Estruturas completas e testadas
- Código modular e reutilizável
- Exemplos de boas práticas
- Comparação de múltiplos algoritmos

## 📂 Arquivos

### 📘 **Estrutura_Machine_Learning.ipynb**

**Descrição**: Template completo demonstrando a estrutura padrão de um projeto de Machine Learning.

**Conteúdo**:

#### 1️⃣ Preparação dos Dados
```python
def criar_dados_classificacao():
    """Cria dataset sintético para classificação"""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y

def preprocessar_dados(X, y, test_size=0.2, scale=True):
    """Preprocessa: split + normalização com stratify automático"""
    # Detecta se é classificação ou regressão
    # Split com stratify se classificação
    # Normalização com StandardScaler
    return X_train, X_test, y_train, y_test, scaler
```

**Features**:
- Geração de dados sintéticos (classificação e regressão)
- Detecção automática de tipo de problema
- Split com estratificação inteligente
- Normalização opcional

---

#### 2️⃣ Algoritmos de Machine Learning

**Classificação** (7 algoritmos):
```python
algorithms = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
```

**Regressão** (3 algoritmos):
```python
algorithms = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}
```

**Estrutura**:
- Classe `MLAlgorithms` para organizar modelos
- Métodos `treinar_classificacao()` e `treinar_regressao()`
- Armazenamento de modelos treinados

---

#### 3️⃣ Otimização de Hiperparâmetros

**GridSearchCV**:
```python
def otimizar_modelo(model, param_grid, X_train, y_train, cv=5):
    """Otimiza hiperparâmetros usando GridSearchCV"""
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search

# Exemplo: Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

best_rf = otimizar_modelo(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    X_train, y_train
)
```

**Output**:
- Melhores parâmetros encontrados
- Melhor score de validação cruzada

---

#### 4️⃣ Avaliação de Modelos

**Métricas de Classificação**:
```python
def avaliar_classificacao(models, X_test, y_test):
    """Avalia múltiplos modelos de classificação"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        }
    
    return pd.DataFrame(results).T
```

**Métricas de Regressão**:
```python
def avaliar_regressao(models, X_test, y_test):
    """Avalia múltiplos modelos de regressão"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        results[name] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R²': r2_score(y_test, y_pred)
        }
    
    return pd.DataFrame(results).T
```

**Output**: DataFrame comparativo com todas as métricas

---

#### 5️⃣ Validação Cruzada

```python
def validacao_cruzada(model, X, y, cv=5):
    """Valida modelo com cross-validation"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return {
        'Mean': scores.mean(),
        'Std': scores.std(),
        'Scores': scores
    }
```

**Uso**: Estimar performance real do modelo

---

#### 6️⃣ Visualizações

**Matriz de Confusão**:
```python
def plotar_confusion_matrix(y_test, y_pred, title=''):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {title}')
    plt.ylabel('Valor Real')
    plt.xlabel('Predição')
    plt.show()
```

**Feature Importance** (Random Forest):
```python
def plotar_feature_importance(model, feature_names):
    """Plota importância das features"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
```

**Comparação de Modelos**:
```python
def plotar_comparacao(results_df, metric='Accuracy'):
    """Plota comparação entre modelos"""
    results_df[metric].plot(kind='barh', figsize=(10, 6))
    plt.title(f'Comparação de Modelos - {metric}')
    plt.xlabel(metric)
    plt.tight_layout()
    plt.show()
```

---

#### 7️⃣ Clustering (K-Means)

```python
def aplicar_kmeans(X, n_clusters=3):
    """Aplica K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    return {
        'labels': clusters,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_
    }

# Elbow Method
def elbow_method(X, max_k=10):
    """Determina número ideal de clusters"""
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo')
    plt.show()
```

---

### 📊 **Compara_Deep.ipynb** (se disponível)

Template para comparação de arquiteturas de Deep Learning (ANN, CNN, RNN).

---

## 🚀 Como Usar

### 1. Copiar o Template

```bash
# Copiar para seu projeto
cp Estrutura_Machine_Learning.ipynb ../MeuProjeto/

# Ou usar como referência
```

### 2. Adaptar ao Seu Problema

**Passo 1**: Substituir geração de dados sintéticos
```python
# Antes:
X, y = criar_dados_classificacao()

# Depois:
df = pd.read_csv('meus_dados.csv')
X = df.drop('target', axis=1)
y = df['target']
```

**Passo 2**: Selecionar algoritmos relevantes
```python
# Manter apenas os modelos que fazem sentido para seu problema
models = {
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()  # Adicionar novos se necessário
}
```

**Passo 3**: Ajustar hiperparâmetros
```python
param_grid = {
    'n_estimators': [100, 200, 500],  # Seus valores
    'max_depth': [10, 20, None]
}
```

**Passo 4**: Executar e avaliar

---

## 📝 Estrutura Padrão de Projeto ML

```
MeuProjeto/
├── data/
│   ├── raw/              # Dados originais
│   └── processed/        # Dados processados
├── notebooks/
│   ├── 01_EDA.ipynb     # Análise exploratória
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb  ← Usar template aqui
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── utils.py
├── models/
│   └── best_model.pkl
├── requirements.txt
└── README.md
```

---

## 🎯 Casos de Uso

### Caso 1: Classificação Binária
```python
# 1. Carregar dados
X, y = load_data()

# 2. Preprocessar
X_train, X_test, y_train, y_test, scaler = preprocessar_dados(X, y)

# 3. Treinar modelos
ml = MLAlgorithms()
models = ml.treinar_classificacao(X_train, y_train)

# 4. Avaliar
results = avaliar_classificacao(models, X_test, y_test)

# 5. Selecionar melhor
best_model = models['Random Forest']
```

### Caso 2: Regressão
```python
# Mesmo fluxo, mas com:
models = ml.treinar_regressao(X_train, y_train)
results = avaliar_regressao(models, X_test, y_test)
```

### Caso 3: Otimização de Hiperparâmetros
```python
# Após treino inicial, otimizar o melhor modelo
param_grid = {...}
best_model = otimizar_modelo(
    RandomForestClassifier(),
    param_grid,
    X_train, y_train
)
```

---

## 📊 Exemplo de Output

### Tabela Comparativa - Classificação
```
                     Accuracy  Precision  Recall  F1-Score
Logistic Regression     0.85       0.84    0.85      0.84
Decision Tree           0.78       0.77    0.78      0.77
Random Forest           0.92       0.91    0.92      0.91
SVM                     0.88       0.87    0.88      0.87
Naive Bayes             0.82       0.81    0.82      0.81
K-Nearest Neighbors     0.86       0.85    0.86      0.85
Gradient Boosting       0.90       0.89    0.90      0.89
```

### Tabela Comparativa - Regressão
```
                      MSE    RMSE     R²
Linear Regression   0.45    0.67   0.82
Decision Tree       0.58    0.76   0.75
Random Forest       0.38    0.62   0.87
```

---

## 💡 Dicas de Uso

### ✅ Boas Práticas

1. **Sempre normalize os dados** para algoritmos sensíveis (KNN, SVM)
2. **Use stratify** no split para classificação
3. **Aplique cross-validation** antes de decidir o melhor modelo
4. **Otimize hiperparâmetros** do modelo escolhido
5. **Salve o modelo treinado** com joblib/pickle

### ⚠️ Armadilhas Comuns

- ❌ Normalizar antes do split (data leakage)
- ❌ Não usar stratify em dados desbalanceados
- ❌ Avaliar apenas no treino (overfit não detectado)
- ❌ Não salvar o scaler junto com o modelo
- ❌ Comparar modelos sem cross-validation

---

## 🔧 Customizações Sugeridas

### Adicionar Novos Algoritmos
```python
# No __init__ da classe MLAlgorithms
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

self.algorithms['classification'].update({
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
})
```

### Adicionar Novas Métricas
```python
from sklearn.metrics import roc_auc_score

def avaliar_com_roc(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
    return pd.DataFrame(results).T
```

### Adicionar Pipeline Completo
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```

---

## 📚 Bibliotecas Utilizadas

```python
# Data
import numpy as np
import pandas as pd

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.datasets import make_classification, make_regression
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

## 🎓 Recursos Adicionais

**Para aprender mais**:
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Kaggle Learn](https://www.kaggle.com/learn)

**Notebooks relacionados**:
- `../Fundamentos/` - Conceitos básicos
- `../Projetos/` - Projetos completos usando este template

---

## ✅ Checklist para Novo Projeto

Ao usar este template em um novo projeto:

- [ ] Copiar `Estrutura_Machine_Learning.ipynb`
- [ ] Substituir dados sintéticos por dados reais
- [ ] Adaptar lista de algoritmos ao problema
- [ ] Definir grid de hiperparâmetros
- [ ] Executar avaliação comparativa
- [ ] Aplicar cross-validation
- [ ] Otimizar melhor modelo
- [ ] Salvar modelo final e preprocessador
- [ ] Documentar resultados

---

**Dúvidas?** Consulte os projetos em `../Projetos/` para ver exemplos completos de uso deste template!

---

*Este template foi criado como parte do projeto "Especialista em IA" - Módulo EAI_02*
