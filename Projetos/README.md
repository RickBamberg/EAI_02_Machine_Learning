# Projetos - Machine Learning

## 📌 Sobre

Esta pasta contém **3 projetos completos** de Machine Learning aplicados a problemas reais. Cada projeto demonstra o pipeline completo desde análise exploratória até modelagem e avaliação.

## 🎯 Objetivo

Aplicar os conceitos aprendidos em **Fundamentos** e **Modelo_Base** em problemas práticos do mundo real, desenvolvendo experiência hands-on em:
- Análise exploratória de dados reais
- Tratamento de problemas específicos de cada domínio
- Escolha e otimização de modelos
- Interpretação de resultados no contexto do problema

---

## 📂 Projetos Disponíveis

### 1️⃣ **Desempenho dos Alunos** 🎓

**Problema**: Prever o desempenho acadêmico de estudantes baseado em fatores demográficos, comportamentais e acadêmicos.

**Tipo**: Classificação Multiclasse (5 classes → 3 classes)

**Dataset**: 
- **Fonte**: Student Performance (UCI ML Repository)
- **Tamanho**: 1,044 registros (Matemática + Português)
- **Features**: 33 variáveis (idade, faltas, consumo de álcool, apoio familiar, etc.)

**Algoritmos Utilizados**:
- Random Forest (padrão e com class_weight)
- Random Forest + SMOTE
- LightGBM
- Otimização com RandomizedSearchCV

**Resultado**: ~58% accuracy com 3 classes

**Lição Principal**: Demonstração honesta de que **nem todo problema tem solução preditiva** com os dados disponíveis. O projeto gerou insights valiosos para políticas educacionais, mas a acurácia preditiva foi limitada pela falta de features críticas.

**Documentação**:
- 📄 [README.md](Desempenho_dos_Alunos/README.md)
- 🤖 [AGENT_CONTEXT.md](Desempenho_dos_Alunos/AGENT_CONTEXT.md)

**Notebooks**:
- `desempenho_dos_alunos.ipynb` - Análise completa e modelagem

---

### 2️⃣ **Detecção de Fraudes** 💳

**Problema**: Detectar transações bancárias fraudulentas em dados extremamente desbalanceados.

**Tipo**: Classificação Binária

**Dataset**: 
- **Fonte**: BankSim (Kaggle - simulador baseado em agentes)
- **Tamanho**: 594,643 transações
- **Desbalanceamento**: 98.79% normal, 1.21% fraude

**Desafio**: Dados altamente desbalanceados requerem técnicas especiais de avaliação (ROC-AUC) e feature engineering extensivo.

**Feature Engineering**: 
- **24 features criadas**:
  - Cliente (12): qtd_transacoes, valor_médio, diversidade de merchants
  - Temporais (4): janela móvel de 5 steps, média e desvio
  - Merchant (1): desvio padrão de valores
  - Relacionamento (3): primeira transação, proporção de uso
  - Localização (2): mesmo CEP, diversidade de CEPs

**Algoritmos Utilizados**:
- Logistic Regression
- Random Forest ⭐ (vencedor)
- Gradient Boosting

**Resultado**: **ROC-AUC 0.98** com Random Forest

**Métricas**:
- Precision: ~90%
- Recall: ~85%
- F1-Score: ~87%

**Lição Principal**: Feature engineering é crucial. As 24 features criadas (especialmente temporais e de relacionamento) foram decisivas para o sucesso do modelo.

**Documentação**:
- 📄 [README.md](Deteccao_Fraudes/README.md)
- 🤖 [AGENT_CONTEXT.md](Deteccao_Fraudes/AGENT_CONTEXT.md)

**Notebooks**:
- `fraud_detection_banksim.ipynb` - Análise e modelagem principal
- `relatorio_para_auditoria.ipynb` - Versão auditável dos resultados

---

### 3️⃣ **Diabetes** 🏥

**Problema**: Prever risco de diabetes baseado em dados clínicos.

**Tipo**: Classificação Binária

**Dataset**: 
- **Fonte**: Pima Indians Diabetes Database (Kaggle)
- **Tamanho**: 768 registros
- **Features**: 8 variáveis (gravidez, glicose, pressão arterial, IMC, idade, etc.)

**Desafio**: Zeros implausíveis (glicose=0, IMC=0) precisaram ser tratados como valores ausentes e imputados.

**Pré-processamento Diferenciado**:
- **Estratégia Mista** (melhor resultado):
  - RobustScaler: Insulina, Espessura da pele, IMC (features com outliers)
  - StandardScaler: Glicose, Pressão arterial (features mais normais)
  - Passthrough: Gravidez, Idade, Diabetes Descendente (já em escala adequada)

**Algoritmos Utilizados**:
- Random Forest (n_estimators=200, max_depth=10)
- Balanceamento com SMOTE (apenas no treino)

**Resultado**: **77-80% accuracy**

**Diferencial**: 
- **Aplicação web completa** com Flask
- **Validação fisiológica** (ranges clinicamente plausíveis)
- **Sistema de logging** robusto
- **Testes automatizados** (/test_model)
- **Executável standalone** (.exe para Windows)

**Deployment**:
```
Diabetes/
├── app.py                    # Aplicação Flask
├── salva_modelo.py          # Script de treino
├── model/
│   ├── model.pkl            # Modelo salvo
│   └── preprocessor.pkl     # Preprocessador
├── templates/               # HTML (index, results, test_results)
├── static/                  # CSS
├── data/
│   └── test_cases.csv       # Casos de teste
└── logs/                    # Sistema de logging
```

**Lição Principal**: Projeto end-to-end completo, desde análise até deployment. Demonstra o ciclo completo de um projeto de ML em produção.

**Documentação**:
- 📄 [README.md](Diabetes/README.md)
- 🤖 [AGENT_CONTEXT.md](Diabetes/AGENT_CONTEXT.md)

**Notebooks**:
- `Diabetes.ipynb` - Análise, pré-processamento e modelagem

**Aplicação**:
- `app.py` - Servidor Flask
- Acesse: `http://localhost:5000`
- Teste: `http://localhost:5000/test_model`

---

## 📊 Comparação dos Projetos

| Projeto | Tipo | Dataset | Resultado | Técnica Destaque |
|---------|------|---------|-----------|------------------|
| **Desempenho Alunos** | Classificação 5→3 classes | 1,044 registros | 58% accuracy | Honestidade sobre limitações |
| **Detecção Fraudes** | Classificação binária | 594k transações | **ROC-AUC 0.98** | **24 features engineered** |
| **Diabetes** | Classificação binária | 768 registros | 77-80% accuracy | **App deployado completo** |

---

## 🎯 O Que Você Aprenderá

### Com Desempenho dos Alunos
- Lidar com dados educacionais
- Tratamento de outliers (faltas extremas)
- Balanceamento de classes com SMOTE
- Reclassificação de target (5 → 3 classes)
- Otimização de hiperparâmetros
- **Reconhecer quando os dados são insuficientes**

### Com Detecção de Fraudes
- Feature engineering avançado
- Dados extremamente desbalanceados (1.21% fraude)
- Importância do ROC-AUC vs Accuracy
- Features temporais (janela móvel)
- Trade-off Precision vs Recall
- Ajuste de threshold por custo de negócio

### Com Diabetes
- Tratamento de zeros implausíveis
- Pré-processamento diferenciado (misto)
- Validação fisiológica de inputs
- **Deployment com Flask**
- Sistema de logging
- Geração de executável
- Testes automatizados

---

## 🚀 Como Usar os Projetos

### Opção 1: Seguir em Ordem

**Recomendado para iniciantes**:
1. Comece com **Diabetes** (mais simples, dataset pequeno)
2. Avance para **Detecção Fraudes** (mais complexo, feature engineering)
3. Finalize com **Desempenho Alunos** (reflexão sobre limitações)

### Opção 2: Por Interesse

**Escolha o projeto que mais te interessa**:
- Saúde → Diabetes
- Finanças → Detecção Fraudes
- Educação → Desempenho Alunos

### Opção 3: Por Habilidade

**Aprender técnica específica**:
- Feature engineering → Detecção Fraudes
- Deployment → Diabetes
- Balanceamento de classes → Desempenho Alunos ou Fraudes

---

## 💡 Dicas de Estudo

### Para Cada Projeto

1. **Leia o README** primeiro para contexto
2. **Execute o notebook** célula por célula
3. **Consulte o AGENT_CONTEXT** para detalhes técnicos
4. **Modifique os parâmetros** e observe o impacto
5. **Tente com seus próprios dados**

### Experimentos Sugeridos

**Desempenho Alunos**:
- Testar outros algoritmos (XGBoost, LightGBM)
- Criar novas features (interações entre variáveis)
- Tentar regressão em vez de classificação

**Detecção Fraudes**:
- Adicionar mais features temporais
- Testar diferentes thresholds
- Calcular custo/benefício de cada threshold

**Diabetes**:
- Comparar diferentes estratégias de pré-processamento
- Adicionar mais validações ao app Flask
- Deploy em cloud (Heroku, Railway)

---

## 📝 Estrutura Comum dos Projetos

Todos os projetos seguem estrutura similar:

```
Projeto/
├── README.md                  # Documentação humana
├── AGENT_CONTEXT.md          # Contexto técnico para IA
├── notebook/
│   └── projeto.ipynb         # Análise e modelagem
├── data/
│   └── dataset.csv           # Dados (se pequeno)
├── model/                    # Modelos salvos (se aplicável)
├── src/                      # Scripts Python (se aplicável)
└── requirements.txt          # Dependências
```

---

## 🔧 Dependências Comuns

```bash
# Instalar para todos os projetos
pip install pandas numpy matplotlib seaborn scikit-learn

# Específico Fraudes
pip install lightgbm

# Específico Diabetes
pip install flask joblib
```

---

## ✅ Checklist de Progresso

### Desempenho dos Alunos
- [ ] Executei o notebook completo
- [ ] Entendi por que a acurácia é baixa
- [ ] Identifiquei os insights educacionais gerados
- [ ] Refleti sobre limitações de dados

### Detecção de Fraudes
- [ ] Entendi as 24 features criadas
- [ ] Compreendi ROC-AUC vs Accuracy
- [ ] Analisei feature importance
- [ ] Pensei em threshold ajustável

### Diabetes
- [ ] Executei o notebook
- [ ] Rodei a aplicação Flask localmente
- [ ] Testei com `/test_model`
- [ ] Entendi validação fisiológica
- [ ] Considerei deployment em cloud

---

## 🎓 Recursos Adicionais

**Datasets Similares**:
- [UCI ML Repository](https://archive.ics.uci.edu/ml/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenML](https://www.openml.org/)

**Para Deployment**:
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Heroku Deploy Guide](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Railway Deploy](https://railway.app/)

**Feature Engineering**:
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Kaggle Feature Engineering](https://www.kaggle.com/learn/feature-engineering)

---

## 🤝 Contribuindo

Quer adicionar seu próprio projeto a esta pasta?

**Critérios**:
1. Dados reais ou realistas
2. Problema bem definido
3. Pipeline completo (EDA → Modelagem → Avaliação)
4. README e AGENT_CONTEXT
5. Código limpo e documentado

---

## 🎯 Próximos Passos

Após completar esses projetos:

1. **Crie seu próprio projeto** usando o template em `../Modelo_Base/`
2. **Participe de competições Kaggle**
3. **Contribua para projetos open source**
4. **Desenvolva um portfólio** de projetos de ML

---

**Dúvidas?** Consulte o AGENT_CONTEXT de cada projeto para detalhes técnicos profundos!

---

*Desenvolvido como parte do projeto "Especialista em IA" - Módulo EAI_02*
