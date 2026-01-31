# Predição de Desempenho de Alunos 🎓

Projeto de Machine Learning para predição de desempenho acadêmico baseado em fatores demográficos, sociais e escolares.

---

## 📌 Sobre o Projeto

Este projeto analisa o desempenho de estudantes usando dados da **Student Performance Database (UCI ML Repository)** e aplica técnicas de Machine Learning para prever notas finais baseadas em variáveis como histórico familiar, hábitos de estudo, consumo de álcool e faltas.

**Objetivo Principal**: Demonstrar que nem todo problema tem solução preditiva com os dados disponíveis, mas ainda assim é possível extrair **insights valiosos** para políticas educacionais.

---

## 🎯 Problema

**Tipo**: Classificação Multiclasse  
**Target Original**: 5 classes (Péssimo, Ruim, Normal, Bom, Excelente)  
**Target Refatorado**: 3 classes (Baixo, Médio, Alto) para melhorar performance

**Desafio**: Prever desempenho acadêmico é complexo porque depende de fatores não capturados nos dados (métodos de ensino, contexto emocional, habilidades cognitivas).

---

## 📊 Dataset

### Fonte
- **Base**: Student Performance Dataset (UCI ML Repository)
- **URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip
- **Arquivos**: 
  - `student-mat.csv` (Matemática - 395 alunos)
  - `student-por.csv` (Português - 649 alunos)
  - **Total Merged**: 1,044 registros

### Características
- **Features**: 33 variáveis
  - Demográficas: sexo, idade, área (urbano/rural)
  - Educacionais: educação dos pais, tempo de estudo, aulas extras
  - Comportamentais: consumo de álcool, tempo livre, saídas
  - Desempenho: faltas, notas G1, G2, G3

### Distribuição da Target (5 classes)
```
Normal (10-13):    43.7%
Ruim (6-9):        24.3%
Bom (14-17):       21.4%
Péssimo (0-5):      7.2%
Excelente (18-20):  3.4%
```

### Distribuição Refatorada (3 classes)
```
Baixo (0-9):   36.4%
Médio (10-13): 35.2%
Alto (14-20):  28.4%
```

---

## 🔧 Metodologia

### 1. Preparação dos Dados
```python
# Merge dos datasets
df_mat = pd.read_csv('student-mat.csv')
df_por = pd.read_csv('student-por.csv')
df_mat['subject'] = 'Matemática'
df_por['subject'] = 'Português'
df = pd.concat([df_mat, df_por], ignore_index=True)
```

### 2. Feature Engineering
```python
# Novas features criadas
df['total_alcool'] = df['Dalc'] + df['Walc']
df['progresso_semestre'] = df['G2'] - df['G1']
df['progresso_final'] = df['G3'] - df['G2']
```

### 3. Tratamento de Data Leakage
**Features removidas** (vazam informação do target):
- `G1`, `G2`, `G3` (notas intermediárias)
- `progresso_semestre`, `progresso_final` (derivadas das notas)

### 4. Tratamento de Outliers
```python
# Remover faltas extremas (> 30 dias)
Q3 = df['absences'].quantile(0.75)
IQR = df['absences'].quantile(0.75) - df['absences'].quantile(0.25)
limite_superior = Q3 + 1.5 * IQR
df = df[df['absences'] <= limite_superior]
```

### 5. Balanceamento (SMOTE)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Antes: 826 amostras
# Depois: 1,368 amostras balanceadas
```

---

## 🤖 Modelos Testados

### 1. Random Forest (Baseline)
```python
RandomForestClassifier(random_state=42)
```
**Resultado**: ~48% accuracy

### 2. Random Forest + Class Weight
```python
RandomForestClassifier(class_weight='balanced', random_state=42)
```
**Resultado**: ~50% accuracy

### 3. Random Forest + SMOTE
```python
RandomForestClassifier(random_state=42)
# Com SMOTE no treino
```
**Resultado**: ~49% accuracy

### 4. LightGBM (5 classes)
```python
LGBMClassifier(random_state=42)
```
**Resultado**: ~50% accuracy

### 5. LightGBM (3 classes) ⭐ MELHOR
```python
LGBMClassifier(random_state=42)
# Com target refatorado para 3 classes
```
**Resultado**: **~58% accuracy**

### 6. Random Forest + RandomizedSearchCV
```python
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=10)
```
**Resultado**: ~52% accuracy

---

## 📈 Resultados Finais

### Melhor Modelo: LightGBM (3 classes)
```
              precision  recall  f1-score  support
Alto              0.54    0.53      0.53       59
Baixo             0.66    0.69      0.68       75
Médio             0.52    0.51      0.51       73

accuracy                            0.58      207
macro avg         0.57    0.58      0.57      207
weighted avg      0.58    0.58      0.58      207
```

**Interpretação**: 
- Accuracy de 58% é **insuficiente** para uso prático em produção
- Classe "Baixo" melhor detectada (66% precision)
- Classe "Médio" mais difícil de prever (52% precision)

---

## 🔍 Análise Exploratória - Insights

### Correlações com Desempenho (G3)
```
Reprovações (failures):     -0.36  (forte negativa)
G2 (nota semestre):          0.90  (muito forte - REMOVIDA)
G1 (nota 1º período):        0.80  (forte - REMOVIDA)
Desejo higher education:     0.18  (fraca positiva)
Tempo de estudo:             0.10  (muito fraca)
Consumo álcool (total):     -0.05  (desprezível)
Faltas (absences):          -0.03  (desprezível)
```

### Insights Importantes

✅ **Fatores que Importam**:
- Histórico de reprovações é o principal preditor
- Aspiração por ensino superior correlaciona positivamente
- Educação dos pais tem impacto moderado

❌ **Fatores com Baixo Impacto** (surpreendente):
- Tempo de estudo tem efeito muito pequeno
- Faltas têm correlação quase zero
- Consumo de álcool tem efeito desprezível

📊 **Diferenças por Matéria**:
- Português tem mais alunos com bom desempenho
- Matemática tem distribuição mais concentrada em notas baixas

---

## 💡 Conclusões e Aprendizados

### ⚠️ Por que a Acurácia é Baixa?

**Causa Raiz**: As features disponíveis **não capturam os principais fatores** que determinam desempenho acadêmico.

**Features Faltantes Críticas**:
- Métodos de ensino do professor
- Qualidade das aulas assistidas (presença ≠ atenção)
- Habilidades cognitivas dos alunos
- Contexto emocional e psicológico
- Motivação e engajamento real
- Suporte pedagógico recebido

### ✅ Valor Gerado pelo Projeto

Apesar da baixa acurácia preditiva, o projeto **gerou insights valiosos**:

1. **Identificação de Grupos de Risco**:
   - Alunos com histórico de reprovações precisam de atenção especial
   - Faltas extremas (>30 dias) são um alerta

2. **Políticas Educacionais**:
   - Reforçar apoio a alunos com pais de baixa escolaridade
   - Incentivar aspiração por ensino superior

3. **Honestidade Científica**:
   - Reconhecer limitações é parte essencial do trabalho com dados
   - Nem todo problema tem solução preditiva com os dados disponíveis

### 🎯 Lição Principal

> **"Às vezes, o valor de um projeto de ML não está em prever o futuro, mas em entender o presente."**

Este projeto é um **excelente exemplo** de quando ser honesto sobre as limitações é mais valioso que forçar uma solução que não funciona.

---

## 🚀 Como Executar

### 1. Configurar Ambiente
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar Notebook
```bash
jupyter notebook desempenho_dos_alunos.ipynb
```

### 3. Estrutura do Código

**Imports principais**:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
```

---

## 📂 Estrutura do Projeto

```
Desempenho_dos_Alunos/
├── README.md                      # Este arquivo
├── AGENT_CONTEXT.md              # Contexto técnico para IA
├── desempenho_dos_alunos.ipynb   # Notebook principal
├── data/
│   ├── student-mat.csv           # Dataset Matemática
│   ├── student-por.csv           # Dataset Português
│   └── student-merged.csv        # Dataset combinado (gerado)
├── models/                        # Modelos salvos (opcional)
└── requirements.txt              # Dependências
```

---

## 📚 Dependências

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
imbalanced-learn>=0.9.0
jupyter>=1.0.0
```

---

## 🔮 Melhorias Futuras

### Para Aumentar Acurácia (se possível)
- [ ] Coletar dados sobre métodos de ensino
- [ ] Adicionar variáveis de habilidades cognitivas
- [ ] Capturar contexto emocional/psicológico
- [ ] Incluir qualidade real da participação nas aulas

### Para Análise
- [ ] Análise de subgrupos específicos
- [ ] Estudo longitudinal (acompanhar mesmos alunos)
- [ ] Comparação entre escolas diferentes
- [ ] Análise de intervenções educacionais

### Técnicas Avançadas
- [ ] Ensemble de múltiplos modelos
- [ ] Deep Learning (MLP) para capturar interações complexas
- [ ] Análise de grafos (redes sociais entre alunos)

---

## 🤝 Contribuindo

Este projeto é educacional e aberto a contribuições!

**Como contribuir**:
1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-analise`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova análise'`)
4. Push para a branch (`git push origin feature/nova-analise`)
5. Abra um Pull Request

---

## 📖 Referências

- Dataset: [UCI ML Repository - Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- Paper Original: P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance" (2008)
- SMOTE: Chawla et al. "SMOTE: Synthetic Minority Over-sampling Technique" (2002)

---

## 📝 Citação

Se usar este projeto, por favor cite:
```
@misc{desempenho_alunos_2026,
  author = {Carlos Henrique Bamberg Marques},
  title = {Predição de Desempenho de Alunos - Análise de Limitações},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/RickBamberg/EAI_02_Machine_Learning}
}
```

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/Portfolio)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

**⚡ Lembre-se**: A honestidade sobre limitações é mais valiosa que acurácia forçada!

*Projeto desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_02*
