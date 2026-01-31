# Fundamentos de Machine Learning

## 📌 Sobre Esta Seção

Esta pasta contém **5 notebooks progressivos** que estabelecem as bases práticas de Machine Learning. Você aprenderá desde análise exploratória de dados até comparação sistemática de algoritmos.

## 🎯 Objetivo

Construir uma compreensão sólida do **pipeline completo de ML**:
1. Analisar e preparar dados
2. Treinar modelos
3. Avaliar performance
4. Comparar algoritmos
5. Tomar decisões baseadas em métricas

## 📚 Notebooks

### 1️⃣ analise_exploratoria.ipynb
**Fundamentos de EDA (Exploratory Data Analysis)**

**O que você aprenderá**:
- Inspecionar estrutura de dados com pandas
- Identificar e tratar valores nulos
- Converter tipos de dados
- Codificar variáveis categóricas
- Preparar dados para ML

**Dataset**: Funcionários (ID, Nome, Idade, Salário, Departamento)

**Tempo estimado**: 30-45 minutos

**Pré-requisitos**: Conhecimento básico de Python e pandas

---

### 2️⃣ classificacao_sintetica.ipynb
**Primeiro Modelo de Classificação**

**O que você aprenderá**:
- Gerar dados sintéticos com `make_classification`
- Visualizar distribuição de classes
- Separar dados em treino e teste
- Normalizar features
- Treinar modelo de classificação
- Avaliar com métricas básicas

**Dataset**: 1000 amostras sintéticas, 2 features, 2 classes

**Algoritmo**: Regressão Logística (baseline simples)

**Tempo estimado**: 45-60 minutos

**Pré-requisitos**: Python, numpy, matplotlib

---

### 3️⃣ classificacao_KNN.ipynb
**K-Nearest Neighbors em Profundidade**

**O que você aprenderá**:
- Como funciona o algoritmo KNN
- Importância da normalização em KNN
- Escolher o valor ideal de `k`
- Visualizar limites de decisão
- Interpretar matriz de confusão
- Fazer predições em novos dados

**Dataset**: 100 amostras sintéticas em 2D

**Conceito principal**: Lazy learning e vizinhança local

**Visualizações**:
- Scatter plot das classes
- Decision boundaries (fronteira de decisão)
- Matriz de confusão

**Tempo estimado**: 1-1.5 horas

**Pré-requisitos**: Compreensão de distância euclidiana

**Dica importante**: Execute com diferentes valores de `k` (3, 5, 7, 11) e observe as mudanças!

---

### 4️⃣ classificacao_projeto_real.ipynb
**ML Aplicado a Dados Médicos**

**O que você aprenderá**:
- Trabalhar com dataset real (569 amostras, 30 features)
- Analisar dados de diagnóstico médico
- Comparar 4 algoritmos diferentes
- Interpretar métricas em contexto médico
- Entender trade-offs entre Precision e Recall

**Dataset**: Wisconsin Breast Cancer
- **Problema**: Classificar tumor como maligno ou benigno
- **Features**: Medições de raio, textura, perímetro, área, etc.
- **Importância**: Recall alto é crucial (detectar todos os casos positivos)

**Modelos testados**:
- Regressão Logística
- Decision Tree
- Random Forest
- KNN

**Tempo estimado**: 1.5-2 horas

**Pré-requisitos**: Notebooks 1-3 completados

**Reflexão importante**: Por que em diagnóstico médico preferimos Recall alto mesmo que isso reduza Precision?

---

### 5️⃣ comparacao_modelos.ipynb
**Comparação Sistemática de Algoritmos**

**O que você aprenderá**:
- Pipeline de comparação de múltiplos modelos
- Treinar 4 algoritmos simultaneamente
- Criar tabela comparativa de métricas
- Visualizar múltiplas matrizes de confusão
- Selecionar o melhor modelo para o problema

**Algoritmos**:
1. Regressão Logística
2. Árvore de Decisão
3. Random Forest
4. KNN

**Output**: Tabela com Accuracy, Precision, Recall, F1 de cada modelo

**Tempo estimado**: 1 hora

**Pré-requisitos**: Todos os notebooks anteriores

**Aplicação prática**: Use este template para qualquer novo projeto de classificação!

---

## 🗂️ Pasta `imagens/`

Contém visualizações salvas pelos notebooks:
- Distribuição de dados sintéticos
- Decision boundaries do KNN
- Matrizes de confusão
- Gráficos comparativos

## 🚀 Como Usar

### Ordem Recomendada
1. **analise_exploratoria.ipynb** - Base de dados
2. **classificacao_sintetica.ipynb** - Primeiro modelo
3. **classificacao_KNN.ipynb** - Algoritmo específico
4. **classificacao_projeto_real.ipynb** - Aplicação real
5. **comparacao_modelos.ipynb** - Visão comparativa

### Executar os Notebooks

```bash
# Entre na pasta
cd EAI_02_Machine_Learning/Fundamentos

# Inicie o Jupyter
jupyter notebook

# Ou JupyterLab
jupyter lab
```

### Experimentação

**Sugestões de modificação** (aprendizado ativo):

📝 **analise_exploratoria.ipynb**:
- Crie seu próprio dataset com `pd.DataFrame()`
- Teste diferentes estratégias de preenchimento de nulos

🔢 **classificacao_sintetica.ipynb**:
- Mude `n_samples` para 500, 2000
- Adicione `n_features=5` e observe o impacto

🎯 **classificacao_KNN.ipynb**:
- Teste `k=1`, `k=3`, `k=15`, `k=50`
- Remova a normalização e veja o resultado

🏥 **classificacao_projeto_real.ipynb**:
- Adicione SVM aos modelos comparados
- Teste diferentes proporções de split (70/30, 60/40)

📊 **comparacao_modelos.ipynb**:
- Adicione Gradient Boosting
- Use dados de outro dataset (Iris, Wine)

## 📖 Conceitos-Chave

### Pipeline de ML
```
Dados → EDA → Preprocessamento → Split → Normalização 
  → Treinamento → Avaliação → Seleção de Modelo
```

### Métricas Essenciais

**Accuracy**: % de acertos totais
```python
accuracy = (TP + TN) / Total
```

**Precision**: Dos que previ como positivo, quantos eram?
```python
precision = TP / (TP + FP)
```

**Recall**: Dos positivos reais, quantos detectei?
```python
recall = TP / (TP + FN)
```

**F1-Score**: Média harmônica de Precision e Recall
```python
f1 = 2 * (precision * recall) / (precision + recall)
```

### Quando Usar Cada Algoritmo

| Situação | Algoritmo Recomendado |
|----------|----------------------|
| Baseline rápido | Regressão Logística |
| Interpretabilidade visual | Decision Tree |
| Máxima acurácia | Random Forest |
| Padrões locais | KNN |
| Muitas features | Random Forest ou SVM |

## ⚠️ Armadilhas Comuns

1. **❌ Normalizar antes de split**: SEMPRE split primeiro!
2. **❌ Usar dados de teste no scaler**: `fit_transform` só no treino
3. **❌ Confiar apenas em Accuracy**: Veja Precision, Recall também
4. **❌ Esquecer random_state**: Resultados não reproduzíveis
5. **❌ KNN sem normalização**: Escala domina a distância

## ✅ Checklist de Progresso

- [ ] Completei análise exploratória e entendo EDA
- [ ] Treinei meu primeiro modelo de classificação
- [ ] Entendo como funciona o KNN
- [ ] Apliquei ML em dados reais
- [ ] Comparei múltiplos algoritmos
- [ ] Sei interpretar matriz de confusão
- [ ] Compreendo trade-off Precision vs Recall
- [ ] Posso escolher o algoritmo adequado para um problema

## 🔗 Próximos Passos

Após completar esta seção:

1. **Explore o Modelo_Base**: Template reutilizável
2. **Escolha um Projeto**: Desempenho_Alunos, Fraudes ou Diabetes
3. **Crie seu próprio projeto**: Aplique o que aprendeu

## 📚 Recursos Adicionais

**Documentação**:
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

**Tutoriais**:
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

**Datasets para Praticar**:
- [UCI Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

**Dúvidas?** Consulte o [AGENT_CONTEXT.md](../AGENT_CONTEXT.md) para detalhes técnicos.

**Próxima etapa**: [Modelo_Base](../Modelo_Base/)
