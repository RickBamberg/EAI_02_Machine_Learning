# AGENT_CONTEXT.md - Desempenho dos Alunos

> **Propósito**: Contexto estruturado para agentes sobre o projeto de análise de desempenho escolar  
> **Última atualização**: Janeiro 2026  
> **Tipo de projeto**: Classificação multiclasse com dados educacionais

## RESUMO EXECUTIVO

**Objetivo**: Prever desempenho acadêmico de alunos baseado em variáveis demográficas, comportamentais e acadêmicas  
**Dataset**: Student Performance (UCI ML Repository)  
**Problema**: Classificação multiclasse (5 classes → 3 classes)  
**Resultado final**: ~50% accuracy (modelo não confiável para predição)  
**Valor**: EDA revelou insights educacionais importantes, pipeline robusto construído

---

## DATASET - CARACTERÍSTICAS

### Fonte
- **URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip
- **Arquivos**: student-mat.csv, student-por.csv
- **Origem**: UC Irvine Machine Learning Repository

### Estrutura
```
df_mat: 395 alunos de Matemática
df_por: 649 alunos de Português
df_alunos (merged): 1044 registros (ambas as matérias)
```

### Features Principais (33 colunas)

**Demográficas**:
- `sex`: Sexo (M/F)
- `age`: Idade (15-22)
- `address`: Tipo de residência (Urban/Rural)
- `famsize`: Tamanho da família (LE3/GT3)
- `Pstatus`: Status dos pais (Together/Apart)

**Educacionais**:
- `Medu`: Educação da mãe (0-4)
- `Fedu`: Educação do pai (0-4)
- `Mjob`: Trabalho da mãe (teacher, health, services, at_home, other)
- `Fjob`: Trabalho do pai
- `reason`: Razão para escolher a escola (home, reputation, course, other)
- `guardian`: Guardião (mother, father, other)

**Acadêmicas**:
- `studytime`: Tempo de estudo semanal (1-4, 1=<2h, 4=>10h)
- `failures`: Número de reprovações anteriores (0-4)
- `schoolsup`: Apoio educacional escolar (yes/no)
- `famsup`: Apoio educacional familiar (yes/no)
- `paid`: Aulas extras pagas (yes/no)
- `activities`: Atividades extracurriculares (yes/no)
- `nursery`: Frequentou creche (yes/no)
- `higher`: Deseja ensino superior (yes/no)

**Comportamentais**:
- `internet`: Acesso à internet em casa (yes/no)
- `romantic`: Em relacionamento romântico (yes/no)
- `famrel`: Qualidade relações familiares (1-5)
- `freetime`: Tempo livre após escola (1-5)
- `goout`: Sair com amigos (1-5)
- `Dalc`: Consumo álcool dia útil (1-5)
- `Walc`: Consumo álcool fim de semana (1-5)
- `health`: Estado de saúde atual (1-5)

**Performance**:
- `absences`: Número de faltas (0-93)
- `G1`: Nota 1º período (0-20)
- `G2`: Nota 2º período (0-20)
- `G3`: Nota final (0-20) - **TARGET original**

**Engineered Features**:
- `subject`: Matéria (Matemática/Português)
- `total_alcool`: Dalc + Walc
- `progresso_semestre`: G2 - G1
- `progresso_final`: G3 - G2
- `desempenho`: Classificação da nota final

---

## VARIÁVEL TARGET

### Target Original (5 classes)
```python
# Baseado em G3 (nota final 0-20)
classificacao = {
    0-5: 'Péssimo',
    6-9: 'Ruim',
    10-13: 'Normal',
    14-17: 'Bom',
    18-20: 'Excelente'
}
```

**Distribuição original**:
```
Normal      456 (43.7%)
Ruim        254 (24.3%)
Bom         223 (21.4%)
Péssimo      75 (7.2%)
Excelente    36 (3.4%)
```

**Problema**: Classes muito desbalanceadas

### Target Refatorado (3 classes)
```python
mapeamento_3_classes = {
    'Péssimo': 'Baixo',      # G3: 0-5
    'Ruim': 'Baixo',         # G3: 6-9
    'Normal': 'Médio',       # G3: 10-13
    'Bom': 'Alto',           # G3: 14-17
    'Excelente': 'Alto'      # G3: 18-20
}
```

**Distribuição refatorada (treino)**:
```
Baixo    301 (36.4%)
Médio    291 (35.2%)
Alto     234 (28.4%)
```

**Distribuição refatorada (teste)**:
```
Baixo    75 (36.2%)
Médio    73 (35.3%)
Alto     59 (28.5%)
```

---

## PIPELINE DO PROJETO

### 1. Carregamento e Merge
```python
df_mat = pd.read_csv("student-mat.csv", sep=';')  # 395 alunos
df_por = pd.read_csv("student-por.csv", sep=';')  # 649 alunos

# Adicionar coluna 'subject'
df_mat['subject'] = 'Matemática'
df_por['subject'] = 'Português'

# Concatenar
df_alunos = pd.concat([df_mat, df_por], ignore_index=True)  # 1044 registros
```

### 2. Feature Engineering
```python
# Total de consumo de álcool
df['total_alcool'] = df['Dalc'] + df['Walc']

# Progresso entre períodos
df['progresso_semestre'] = df['G2'] - df['G1']
df['progresso_final'] = df['G3'] - df['G2']

# Classificação de desempenho (target)
def classificar_nota(nota):
    if nota <= 5: return 'Péssimo'
    elif nota <= 9: return 'Ruim'
    elif nota <= 13: return 'Normal'
    elif nota <= 17: return 'Bom'
    else: return 'Excelente'

df['desempenho'] = df['G3'].apply(classificar_nota)
```

### 3. Tratamento de Outliers
```python
# Remoção de faltas extremas (> 30)
Q3 = df['absences'].quantile(0.75)
IQR = df['absences'].quantile(0.75) - df['absences'].quantile(0.25)
limite_superior = Q3 + 1.5 * IQR  # ~30 faltas

df = df[df['absences'] <= limite_superior]
```

**Justificativa**: Faltas > 30 são outliers extremos que distorcem o modelo

### 4. Remoção de Features (Data Leakage)
```python
# Variáveis que causam vazamento de informação
colunas_drop = ['G1', 'G2', 'G3', 'progresso_semestre', 'progresso_final']

X = df.drop(columns=colunas_drop + ['desempenho'])
y = df['desempenho']
```

**Razão**: G1, G2, G3 são notas parciais que predizem perfeitamente a nota final

### 5. Split Treino/Teste
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Manter proporção de classes
)

# Resultado:
# Treino: 826 amostras (80%)
# Teste: 207 amostras (20%)
```

### 6. Encoding de Variáveis Categóricas
```python
# One-Hot Encoding
categorical_cols = X_train.select_dtypes(include=['object']).columns

X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Alinhar colunas (caso teste tenha categorias diferentes)
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, 
    join='left', 
    axis=1, 
    fill_value=0
)
```

**Resultado**: ~38 features após encoding

### 7. Normalização
```python
from sklearn.preprocessing import StandardScaler

# Identificar colunas numéricas
numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
                'failures', 'famrel', 'freetime', 'goout', 
                'Dalc', 'Walc', 'health', 'absences', 'total_alcool']

scaler = StandardScaler()
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_encoded[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test_encoded[numeric_cols])
```

### 8. Balanceamento com SMOTE
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Antes: 826 amostras desbalanceadas
# Depois: ~1368 amostras balanceadas (456 de cada classe)
```

---

## MODELOS TESTADOS

### 1. Random Forest Padrão
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
# Resultado: ~48%
```

### 2. Random Forest com class_weight
```python
rf_balanced = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
rf_balanced.fit(X_train_scaled, y_train)

# Resultado: ~50%
```

### 3. Random Forest com SMOTE
```python
rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_balanced, y_train_balanced)

# Resultado: ~49%
```

### 4. LightGBM
```python
import lightgbm as lgb

lgbm = lgb.LGBMClassifier(random_state=42)
lgbm.fit(X_train_scaled, y_train)

# Resultado (5 classes): ~50%
```

### 5. LightGBM com 3 Classes
```python
# Reclassificar target
y_train_3c = y_train.map(mapeamento_3_classes)
y_test_3c = y_test.map(mapeamento_3_classes)

lgbm_3c = lgb.LGBMClassifier(random_state=42)
lgbm_3c.fit(X_train_scaled, y_train_3c)

# Resultado: ~58%
```

**Classification Report**:
```
              precision    recall  f1-score   support

        Alto       0.54      0.53      0.53        59
       Baixo       0.66      0.69      0.68        75
       Médio       0.52      0.51      0.51        73

    accuracy                           0.58       207
   macro avg       0.57      0.58      0.57       207
weighted avg       0.58      0.58      0.58       207
```

### 6. Otimização de Hiperparâmetros
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train_balanced, y_train_balanced)

# Melhores parâmetros encontrados:
# n_estimators: 200
# max_depth: 20
# min_samples_split: 10
# min_samples_leaf: 2

# Resultado: ~52%
```

---

## ANÁLISE EXPLORATÓRIA - PRINCIPAIS INSIGHTS

### 1. Correlações com Desempenho
```
Features mais correlacionadas com G3 (nota final):
1. failures (reprovações anteriores): -0.36  ← NEGATIVA
2. G2 (nota 2º período): +0.90             ← FORTE (mas removida por leakage)
3. G1 (nota 1º período): +0.80             ← FORTE (mas removida por leakage)
4. absences (faltas): -0.03                ← FRACA
5. studytime: +0.10                        ← FRACA
```

**Conclusão**: Sem G1/G2, nenhuma feature tem correlação forte com desempenho

### 2. Desempenho por Matéria
```
Matemática:
  Péssimo: 8.6%
  Ruim: 27.8%
  Normal: 48.9%
  Bom: 12.9%
  Excelente: 1.8%

Português:
  Péssimo: 6.5%
  Ruim: 22.5%
  Normal: 41.0%
  Bom: 25.6%
  Excelente: 4.3%
```

**Insight**: Português tem mais alunos em categorias altas

### 3. Impacto do Consumo de Álcool
```
Total_alcool baixo (2-4): Média G3 = 11.2
Total_alcool médio (5-6): Média G3 = 10.8
Total_alcool alto (7-10): Média G3 = 10.3
```

**Insight**: Relação negativa, mas fraca

### 4. Apoio Familiar e Desempenho
```
Com apoio familiar:
  Média G3: 11.1

Sem apoio familiar:
  Média G3: 10.5
```

**Insight**: Apoio familiar ajuda, mas impacto limitado

### 5. Faltas e Desempenho
```
Faltas 0-5: Média G3 = 11.2
Faltas 6-15: Média G3 = 10.4
Faltas 16+: Média G3 = 9.8
```

**Insight**: Mais faltas → piores notas (esperado)

### 6. Ensino Superior
```
Deseja ensino superior (higher=yes):
  Média G3: 11.5

Não deseja (higher=no):
  Média G3: 9.2
```

**Insight**: Aspirações acadêmicas correlacionam com desempenho

---

## TESTES ESTATÍSTICOS REALIZADOS

### 1. Chi-Quadrado (Categóricas vs Categóricas)
```python
from scipy.stats import chi2_contingency

# Exemplo: Sexo vs Desempenho
contingency = pd.crosstab(df['sex'], df['desempenho'])
chi2, p_value, dof, expected = chi2_contingency(contingency)

# p-value < 0.05 → associação significativa
```

**Resultado**: Sexo tem associação fraca com desempenho

### 2. ANOVA (Numérica vs Categórica)
```python
from scipy.stats import f_oneway

# Exemplo: Nota final por nível de consumo de álcool
grupos = [df[df['total_alcool'] == i]['G3'] for i in range(2, 11)]
f_stat, p_value = f_oneway(*grupos)

# p-value < 0.05 → diferença significativa entre grupos
```

### 3. Teste t (Binária vs Numérica)
```python
from scipy.stats import ttest_ind

# Exemplo: Nota de quem tem vs não tem internet
with_internet = df[df['internet'] == 'yes']['G3']
without_internet = df[df['internet'] == 'no']['G3']

t_stat, p_value = ttest_ind(with_internet, without_internet)
```

---

## RESULTADOS FINAIS

### Métricas dos Modelos

| Modelo | Classes | Accuracy | Observações |
|--------|---------|----------|-------------|
| Random Forest | 5 | ~48% | Baseline |
| RF + class_weight | 5 | ~50% | Sem melhora significativa |
| RF + SMOTE | 5 | ~49% | Balanceamento não ajudou |
| LightGBM | 5 | ~50% | Similar ao RF |
| LightGBM | 3 | **~58%** | Melhor resultado |
| RF Otimizado | 5 | ~52% | GridSearch não resolveu |

### Interpretação dos Resultados

**Por que 50% de accuracy?**

1. **Features insuficientes**: Variáveis demográficas/comportamentais têm baixa correlação com desempenho
2. **Ausência de dados críticos**: Faltam informações sobre:
   - Métodos pedagógicos
   - Contexto emocional do aluno
   - Habilidades cognitivas
   - Presença efetiva em aulas
   - Qualidade do estudo (não apenas quantidade)

3. **Vazamento de informação removido**: G1, G2 predizem G3 perfeitamente, mas foram removidas (corretamente) por leakage

4. **Natureza do problema**: Desempenho acadêmico é multifatorial e complexo

---

## CONCLUSÕES DO PROJETO

### ✅ Valor Gerado
1. **Pipeline robusto**: Estrutura completa de ML educacional
2. **Insights educacionais**:
   - Faltas são prejudiciais
   - Reprovações anteriores são forte indicador
   - Aspirações acadêmicas importam
   - Apoio familiar tem impacto positivo

3. **Demonstração de limites**: Nem todo problema tem solução preditiva com dados disponíveis

### ❌ Limitações Identificadas
1. Dataset insuficiente para predição confiável
2. Acurácia de 50-58% é inadequada para uso prático
3. Modelos não generalizam bem

### 🎯 Aplicações Práticas
Apesar da baixa acurácia preditiva, o projeto pode:
- **Orientar políticas educacionais**: Foco em alunos com muitas faltas
- **Identificar grupos de risco**: Alunos sem apoio familiar
- **Análise descritiva**: Entender padrões de desempenho

### 🔮 Próximos Passos Sugeridos
1. **Coletar mais dados**:
   - Histórico de anos anteriores
   - Avaliações formativas contínuas
   - Dados qualitativos (questionários)

2. **Mudar o problema**:
   - Em vez de predição, usar clustering para perfis de alunos
   - Regressão para prever nota contínua (em vez de classes)
   - Detecção de anomalias para alunos em risco

3. **Feature engineering avançada**:
   - Interações entre variáveis
   - Features temporais (evolução ao longo dos períodos)

---

## CÓDIGO DE REFERÊNCIA - SNIPPETS PRINCIPAIS

### Pipeline Completo Resumido
```python
# 1. Carregamento
df_mat = pd.read_csv("student-mat.csv", sep=';')
df_por = pd.read_csv("student-por.csv", sep=';')
df = pd.concat([df_mat, df_por])

# 2. Feature Engineering
df['total_alcool'] = df['Dalc'] + df['Walc']
df['desempenho'] = df['G3'].apply(classificar_nota)

# 3. Tratamento de Outliers
df = df[df['absences'] <= 30]

# 4. Separação X/y (removendo leakage)
X = df.drop(columns=['G1', 'G2', 'G3', 'desempenho', 'progresso_semestre', 'progresso_final'])
y = df['desempenho']

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 6. Encoding
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 7. Scaling
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 8. Balanceamento (opcional)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 9. Modelagem
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 10. Avaliação
print(classification_report(y_test, y_pred))
```

---

## BIBLIOTECAS UTILIZADAS

```python
# Data
import pandas as pd
import numpy as np

# Visualização
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Balanceamento
from imblearn.over_sampling import SMOTE

# Modelos
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# Avaliação
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Otimização
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Testes estatísticos
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
```

---

## PERGUNTAS FREQUENTES - RESPOSTAS PARA AGENTES

**Q: Por que a accuracy ficou em ~50%?**
A: Dataset não tem features suficientes para predição confiável. Variáveis demográficas/comportamentais têm baixa correlação com desempenho. Após remover G1/G2 (leakage), restam features fracas.

**Q: Por que remover G1 e G2?**
A: Data leakage. G1 e G2 são notas parciais que compõem G3 (nota final). Usar elas no treino seria trapaça, pois na prática não teríamos essas notas antes do resultado final.

**Q: SMOTE melhorou o modelo?**
A: Não significativamente (~1-2% de ganho). O problema não é apenas desbalanceamento, mas falta de features preditivas.

**Q: Por que 3 classes funcionou melhor que 5?**
A: Reduzir de 5 para 3 classes simplifica o problema, reduzindo a granularidade e facilitando a separação. Mas ainda assim 58% é baixo.

**Q: O projeto foi um fracasso?**
A: Não. Demonstrou que nem todo problema tem solução preditiva com dados disponíveis. Gerou insights valiosos e um pipeline robusto que pode ser reutilizado com dados melhores.

**Q: Como melhorar os resultados?**
A: (1) Coletar mais dados (anos anteriores, avaliações formativas), (2) Mudar o problema (clustering em vez de classificação), (3) Feature engineering mais avançada, (4) Usar dados qualitativos.

**Q: Qual o melhor modelo testado?**
A: LightGBM com 3 classes (58% accuracy). Mas nenhum modelo atingiu acurácia aceitável para uso prático (>70%).

---

## TAGS DE BUSCA

`#classificacao-multiclasse` `#dados-educacionais` `#student-performance` `#smote` `#balanceamento` `#lightgbm` `#random-forest` `#feature-engineering` `#data-leakage` `#eda` `#analise-exploratoria` `#pipeline-ml` `#low-accuracy` `#limitacoes-ml`

---

**Versão**: 1.0  
**Compatibilidade**: Agentes de IA com conhecimento de ML, classificação e análise de dados  
**Uso recomendado**: Responder perguntas sobre o projeto, metodologia, resultados ou limitações
