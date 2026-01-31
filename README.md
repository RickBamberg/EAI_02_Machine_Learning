# EAI_02 - Machine Learning

## 📚 Sobre este Módulo

Este módulo introduz os **fundamentos práticos de Machine Learning**, aplicando algoritmos clássicos de aprendizado supervisionado em problemas reais de classificação e regressão. Aqui você construirá uma base sólida em ML antes de avançar para Deep Learning.

## 🎯 Objetivos de Aprendizagem

Ao finalizar este módulo, você será capaz de:

- ✅ Realizar análise exploratória de dados (EDA)
- ✅ Implementar e avaliar algoritmos de classificação
- ✅ Aplicar técnicas de pré-processamento e normalização
- ✅ Comparar múltiplos modelos de ML
- ✅ Desenvolver pipelines completos de ML
- ✅ Interpretar métricas de desempenho (Acurácia, Precision, Recall, F1)
- ✅ Utilizar validação cruzada e otimização de hiperparâmetros

## 📂 Estrutura do Módulo

```
EAI_02_Machine_Learning/
├── README.md                          ← Este arquivo
├── AGENT_CONTEXT.md                   ← Contexto técnico
│
├── Fundamentos/
│   ├── README.md                      ← Documentação dos fundamentos
│   ├── analise_exploratoria.ipynb     ← EDA com pandas
│   ├── classificacao_sintetica.ipynb  ← Dados artificiais
│   ├── classificacao_KNN.ipynb        ← K-Nearest Neighbors
│   ├── classificacao_projeto_real.ipynb ← Dataset câncer de mama
│   ├── comparacao_modelos.ipynb       ← 4 algoritmos comparados
│   └── imagens/                       ← Visualizações salvas
│
├── Modelo_Base/
│   ├── README.md                      ← Template de referência
│   └── Estrutura_Machine_Learning.ipynb ← Pipeline completo
│
└── Projetos/
    ├── README.md                      ← Índice dos projetos
    │
    ├── Desempenho_dos_Alunos/         ← Projeto completo
    │   ├── README.md
    │   ├── AGENT_CONTEXT.md
    │   └── [arquivos do projeto...]
    │
    ├── Deteccao_Fraudes/              ← Projeto completo
    │   ├── README.md
    │   ├── AGENT_CONTEXT.md
    │   └── [arquivos do projeto...]
    │
    └── Diabetes/                      ← Projeto com deployment
        ├── README.md
        ├── AGENT_CONTEXT.md
        └── [arquivos do projeto...]
```

## 📖 Conteúdo Detalhado

### 🔹 Fundamentos

Esta seção contém 5 notebooks progressivos que cobrem os conceitos essenciais de Machine Learning.

#### **1. analise_exploratoria.ipynb**
**Objetivo**: Dominar análise exploratória de dados (EDA)

**Tópicos cobertos**:
- Importação e inspeção de datasets
- Estatísticas descritivas (`df.describe()`, `df.info()`)
- Identificação e tratamento de valores nulos
- Conversão de tipos de dados
- Codificação de variáveis categóricas (Label Encoding, One-Hot)
- Preenchimento de dados faltantes (média, mediana, moda)
- Visualizações com matplotlib

**Dataset**: Dados simulados de funcionários (ID, Nome, Idade, Salário, Departamento)

**Bibliotecas**: pandas, numpy, matplotlib

---

#### **2. classificacao_sintetica.ipynb**
**Objetivo**: Introduzir classificação com dados controlados

**Tópicos cobertos**:
- Geração de dados sintéticos (`make_classification`)
- Visualização de distribuição de classes
- Separação treino/teste (70/30)
- Normalização com `StandardScaler`
- Treinamento de modelo básico
- Métricas de avaliação

**Dataset**: 1000 amostras, 2 features, 2 classes

**Algoritmo**: Regressão Logística (simples e interpretável)

---

#### **3. classificacao_KNN.ipynb**
**Objetivo**: Explorar o algoritmo K-Nearest Neighbors

**Tópicos cobertos**:
- Funcionamento do KNN (vizinhos mais próximos)
- Efeito do parâmetro `k` (número de vizinhos)
- Importância da normalização no KNN
- Matriz de confusão
- Visualização de limites de decisão (decision boundaries)
- Predição de novos pontos

**Dataset**: Dados sintéticos 2D (100 amostras, 2 features)

**Visualizações**:
- Scatter plot das classes
- Mapa de calor dos limites de decisão
- Fronteira de decisão do KNN

**Considerações sobre KNN**:
- Sensível à escala dos dados
- Computacionalmente custoso em datasets grandes
- Não há "treinamento" real (lazy learning)

---

#### **4. classificacao_projeto_real.ipynb**
**Objetivo**: Aplicar ML em problema real de saúde

**Tópicos cobertos**:
- Carregamento de dataset real (sklearn.datasets)
- Análise exploratória de dados médicos
- Pré-processamento de 30 features
- Comparação de múltiplos modelos
- Interpretação de resultados médicos

**Dataset**: Wisconsin Breast Cancer
- 569 amostras
- 30 features numéricas (raio, textura, perímetro, área, etc.)
- 2 classes: Maligno (0) / Benigno (1)

**Modelos testados**:
- Regressão Logística
- Decision Tree
- Random Forest
- KNN

**Métricas**:
- Acurácia
- Precision (crucial em diagnóstico médico)
- Recall (detectar todos os casos positivos)
- F1-Score

---

#### **5. comparacao_modelos.ipynb**
**Objetivo**: Comparar sistematicamente algoritmos de ML

**Tópicos cobertos**:
- Pipeline de comparação de modelos
- Treinamento paralelo de 4 algoritmos
- Tabela comparativa de métricas
- Visualização de matrizes de confusão lado a lado
- Escolha do melhor modelo baseado em critérios

**Algoritmos comparados**:
1. **Regressão Logística** - Linear, rápido, interpretável
2. **Árvore de Decisão** - Não-linear, visual, propenso a overfit
3. **Random Forest** - Ensemble, robusto, menos overfit
4. **KNN** - Baseado em instâncias, sem assunções

**Dataset**: 1000 amostras sintéticas, 2 features

**Output**: Tabela resumo com Accuracy, Precision, Recall, F1 de cada modelo

---

### 🔹 Modelo_Base

#### **Estrutura_Machine_Learning.ipynb**
**Objetivo**: Template reutilizável para projetos de ML

**Estrutura completa**:

1. **Preparação dos Dados**
   - Funções para criar datasets sintéticos
   - Preprocessamento automático
   - Normalização e split treino/teste

2. **Modelos de Classificação**
   - Regressão Logística
   - Decision Tree
   - Random Forest
   - SVM
   - Naive Bayes
   - KNN
   - Gradient Boosting

3. **Modelos de Regressão**
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor

4. **Avaliação de Modelos**
   - Funções genéricas para classificação e regressão
   - Cross-validation
   - Matriz de confusão
   - Métricas customizadas

5. **Otimização de Hiperparâmetros**
   - GridSearchCV
   - Busca exaustiva de melhores parâmetros

6. **Clustering (Não-Supervisionado)**
   - K-Means
   - Elbow method

**Uso**: Copie e adapte este notebook para novos projetos

---

### 🔹 Projetos

Três projetos completos aplicando ML em problemas reais:

#### **1. Desempenho_dos_Alunos**
Predição de desempenho acadêmico baseado em fatores socioeconômicos.

**Ver**: [Projetos/Desempenho_dos_Alunos/README.md](Projetos/Desempenho_dos_Alunos/README.md)

---

#### **2. Deteccao_Fraudes**
Sistema de detecção de transações fraudulentas em dados bancários.

**Ver**: [Projetos/Deteccao_Fraudes/README.md](Projetos/Deteccao_Fraudes/README.md)

---

#### **3. Diabetes**
Aplicação web para predição de diabetes com modelo deployado.

**Ver**: [Projetos/Diabetes/README.md](Projetos/Diabetes/README.md)

---

## 🚀 Como Usar Este Módulo

### Pré-requisitos

```bash
# Bibliotecas principais
pip install numpy pandas matplotlib seaborn
pip install scikit-learn

# Para projetos específicos, consulte requirements.txt de cada pasta
```

### Ordem Recomendada de Estudo

**Fase 1 - Fundamentos (1-2 semanas)**
1. `analise_exploratoria.ipynb` - Base de EDA
2. `classificacao_sintetica.ipynb` - Primeiro modelo
3. `classificacao_KNN.ipynb` - Algoritmo intuitivo
4. `classificacao_projeto_real.ipynb` - Dados reais
5. `comparacao_modelos.ipynb` - Visão comparativa

**Fase 2 - Template (3-4 dias)**
6. `Estrutura_Machine_Learning.ipynb` - Pipeline completo

**Fase 3 - Projetos (2-3 semanas)**
7. Escolha um projeto do seu interesse
8. Complete o projeto do início ao fim
9. Experimente com seus próprios dados

### Executando os Notebooks

```bash
# Entre no diretório
cd EAI_02_Machine_Learning/Fundamentos

# Inicie o Jupyter
jupyter notebook

# Ou use JupyterLab
jupyter lab
```

## 💡 Conceitos-Chave Aprendidos

### Análise Exploratória (EDA)
- **Objetivo**: Conhecer os dados antes de modelar
- **Ferramentas**: `df.head()`, `df.info()`, `df.describe()`
- **Importância**: 80% do trabalho de ML é preparação de dados

### Classificação vs Regressão
- **Classificação**: Prever categorias (Spam/Não-spam, Doente/Saudável)
- **Regressão**: Prever valores contínuos (Preço, Temperatura, Salário)

### Treinamento vs Teste
- **Treino**: Modelo aprende padrões (70-80% dos dados)
- **Teste**: Avaliar generalização (20-30% dos dados)
- **Regra de ouro**: NUNCA usar dados de teste no treino

### Normalização
- **Por quê?**: Algoritmos como KNN e SVM são sensíveis à escala
- **StandardScaler**: (x - média) / desvio_padrão
- **Quando**: Fit no treino, transform no teste

### Métricas de Avaliação

**Para Classificação**:
- **Acurácia**: % de acertos totais
- **Precision**: De todos que previ como positivo, quantos eram?
- **Recall**: De todos os positivos reais, quantos detectei?
- **F1-Score**: Média harmônica de Precision e Recall

**Para Regressão**:
- **MSE**: Erro quadrático médio
- **RMSE**: Raiz do MSE (mesma unidade da variável)
- **R²**: Percentual de variância explicada

### Overfitting vs Underfitting
- **Overfitting**: Modelo decora o treino, falha no teste
- **Underfitting**: Modelo muito simples, não captura padrões
- **Solução**: Validação cruzada, regularização, mais dados

### Algoritmos Principais

| Algoritmo | Tipo | Vantagens | Desvantagens |
|-----------|------|-----------|--------------|
| Regressão Logística | Linear | Rápido, interpretável | Assume linearidade |
| KNN | Instance-based | Simples, sem assunções | Lento, sensível a escala |
| Decision Tree | Não-linear | Interpretável, visual | Overfit fácil |
| Random Forest | Ensemble | Robusto, acurado | Black box, lento |
| SVM | Margin-based | Bom em alta dimensão | Difícil de tunear |
| Naive Bayes | Probabilístico | Rápido, bom com texto | Assume independência |

## 🔗 Conexão com Outros Módulos

### De EAI_01 (Fundamentos Matemáticos)
- **Vetores**: Features são vetores em espaço multidimensional
- **Distância euclidiana**: Base do KNN
- **Regressão linear**: Primeiro modelo de ML
- **Álgebra linear**: Transformações e projeções

### Para EAI_03 (Deep Learning)
- **Pipeline de ML**: Mesma estrutura (dados → modelo → avaliação)
- **Normalização**: Crucial em redes neurais
- **Métricas**: Mesmas para avaliar performance
- **Overfitting**: Problema ainda maior em DL

### Para EAI_04 (NLP)
- **Classificação de texto**: Usar algoritmos deste módulo
- **Features numéricas**: TF-IDF → vetor → ML clássico
- **Baseline**: Sempre começar com ML clássico antes de Transformers

### Para EAI_08 (MLOps)
- **Projetos deste módulo**: Serão deployados em EAI_08
- **Pipeline estruturado**: Base para produção
- **Versionamento de modelos**: Preparação para MLflow

## 📝 Notas Importantes

### Boas Práticas
1. **Sempre separe treino/teste** antes de qualquer análise
2. **Normalize os dados** quando necessário
3. **Use validação cruzada** para métricas confiáveis
4. **Comece simples**: Regressão Logística antes de Deep Learning
5. **Visualize os dados**: Gráficos revelam padrões escondidos

### Armadilhas Comuns
- ❌ Usar dados de teste no treinamento (data leakage)
- ❌ Esquecer de normalizar
- ❌ Não verificar valores nulos
- ❌ Ignorar desbalanceamento de classes
- ❌ Confiar cegamente na acurácia

### Dicas de Estudo
- 💡 Execute cada célula e **mude os parâmetros**
- 💡 Tente com **seus próprios dados**
- 💡 Compare resultados com **diferentes splits**
- 💡 Leia a **documentação do sklearn**
- 💡 Participe de **competições Kaggle**

## 🎓 Recursos Complementares

### Cursos Online
- **Coursera**: "Machine Learning" - Andrew Ng
- **Fast.ai**: "Introduction to Machine Learning for Coders"
- **Google**: "Machine Learning Crash Course"

### Livros
- "Hands-On Machine Learning" - Aurélien Géron
- "Introduction to Statistical Learning" - James, Witten, Hastie, Tibshirani
- "Pattern Recognition and Machine Learning" - Christopher Bishop

### Documentação
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

### Datasets para Praticar
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenML](https://www.openml.org/)

## ✅ Checklist de Progresso

### Fundamentos
- [ ] Completei análise exploratória
- [ ] Entendi classificação com dados sintéticos
- [ ] Implementei KNN do zero
- [ ] Apliquei ML em dados reais
- [ ] Comparei múltiplos modelos

### Modelo Base
- [ ] Explorei o template completo
- [ ] Adaptei para meu próprio problema

### Projetos
- [ ] Completei pelo menos 1 projeto
- [ ] Explorei os 3 projetos disponíveis
- [ ] Criei meu próprio projeto de ML

### Conceitos
- [ ] Entendo quando usar cada algoritmo
- [ ] Sei interpretar matriz de confusão
- [ ] Conheço as principais métricas
- [ ] Compreendo overfitting e como evitá-lo

## 🤝 Contribuindo

Encontrou um erro ou tem uma sugestão? Abra uma issue ou envie um pull request!

---

**Próximo Módulo**: [EAI_03 - Deep Learning](../EAI_03_Deep_Learning)

**Anterior**: [EAI_01 - Fundamentos Matemáticos](../EAI_01_Fundamentos_Matemática_para_IA)

---

*Desenvolvido como parte do projeto "Especialista em IA"*
