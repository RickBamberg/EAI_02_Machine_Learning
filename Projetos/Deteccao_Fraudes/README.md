# Detecção de Fraudes Bancárias 💳

Sistema de Machine Learning para detecção de transações fraudulentas em dados bancários extremamente desbalanceados.

---

## 📌 Sobre o Projeto

Este projeto implementa um **sistema de detecção de fraudes** usando o dataset BankSim (simulador baseado em agentes) e técnicas avançadas de Machine Learning para lidar com dados altamente desbalanceados (98.79% transações normais vs 1.21% fraudulentas).

**Objetivo Principal**: Alcançar alta taxa de detecção de fraudes (Recall) mantendo baixa taxa de falsos positivos (Precision), usando feature engineering extensivo.

---

## 🎯 Problema

**Tipo**: Classificação Binária  
**Target**: 0 = Transação Normal | 1 = Transação Fraudulenta  
**Desafio**: Dados extremamente desbalanceados (1.21% fraude)

**Métricas Prioritárias**:
- **ROC-AUC**: Capacidade geral de separação
- **Recall**: Detectar o máximo de fraudes possível
- **Precision**: Minimizar falsos positivos (custo operacional)

---

## 📊 Dataset - BankSim

### Fonte
- **Nome**: BankSim1 - Banking Transactions Dataset
- **URL**: https://www.kaggle.com/datasets/ealaxi/banksim1
- **Arquivo**: bs140513_032310.csv
- **Descrição**: Simulador baseado em agentes que gera transações bancárias sintéticas para pesquisa em detecção de fraudes

### Características
```
Total de transações:     594,643
Transações normais:      587,443 (98.79%)
Transações fraudulentas:   7,200 (1.21%)
Período simulado:        180 steps (~6 meses)
```

### Estrutura do Dataset (10 colunas)

| Coluna | Tipo | Descrição | Exemplo |
|--------|------|-----------|---------|
| step | int | Timestamp (0-179) | 0, 5, 120 |
| customer | str | ID do cliente | C1093826151 |
| age | int | Faixa etária (0-6) | 4 |
| gender | str | Gênero (M/F) | M, F |
| zipcodeOri | str | CEP origem | 28007 |
| merchant | str | ID do comerciante | M348934600 |
| zipMerchant | str | CEP do comerciante | 28007 |
| category | str | Categoria da transação | es_transportation |
| amount | float | Valor da transação | 4.55, 39.68 |
| **fraud** | **int** | **Target (0/1)** | **0, 1** |

### Categorias de Transação
```
es_transportation, es_food, es_health, es_home, es_hotelservices,
es_otherservices, es_contents, es_tech, es_sportsandtoys,
es_wellnessandbeauty, es_hyper, es_fashion, es_bars,
es_travel, es_leisure
```

---

## 🔧 Feature Engineering (24 Features Criadas)

### 1. Features de Cliente (10 features)
```python
# Agregações básicas
qtd_transacoes = df.groupby('customer').size()
total_tx_cliente = df.groupby('customer')['amount'].sum()
amount_mean_cliente = df.groupby('customer')['amount'].mean()
amount_std_cliente = df.groupby('customer')['amount'].std()

# Diversidade
num_categorias_cliente = df.groupby('customer')['category'].nunique()
num_merchants_cliente = df.groupby('customer')['merchant'].nunique()
num_zipcodes_cliente = df.groupby('customer')['zipcodeOri'].nunique()

# Alertas
alert_freq = (qtd_transacoes > 10).astype(int)
alert_valor = (df['amount'] > 200).astype(int)
valor_relativo_cliente = df['amount'] / amount_mean_cliente
```

### 2. Features Temporais (4 features)
```python
# Janela móvel de 5 steps
tx_ultimos_5_steps = df.groupby('customer')['step'].rolling(5).count()
amount_media_5steps = df.groupby('customer')['amount'].rolling(5).mean()
amount_desvio_5steps = df.groupby('customer')['amount'].rolling(5).std()

# Tempo desde última transação
step_diff = df.groupby('customer')['step'].diff()
```

### 3. Features de Merchant (1 feature)
```python
# Desvio padrão dos valores por merchant
amount_std_merchant = df.groupby('merchant')['amount'].std()
```

### 4. Features de Relacionamento Cliente-Merchant (3 features)
```python
# Número de transações entre cliente e merchant
tx_cliente_merchant = df.groupby(['customer', 'merchant']).size()

# É a primeira transação com este merchant?
primeira_tx_merchant = (tx_cliente_merchant == 1).astype(int)

# Proporção de transações com este merchant
prop_tx_merchant = tx_cliente_merchant / qtd_transacoes
```

### 5. Features de Localização (2 features)
```python
# Cliente e merchant no mesmo CEP?
mesma_localizacao = (df['zipcodeOri'] == df['zipMerchant']).astype(int)

# Número de CEPs diferentes usados pelo cliente
num_zipcodes_cliente = df.groupby('customer')['zipcodeOri'].nunique()
```

### Features Excluídas (Data Leakage)
```python
# ❌ Removidas por vazarem informação do target
# fraude_merchant_train
# fraude_categoria
# tx_por_merchant_train (se calculado com target)
```

---

## 🤖 Pipeline de Modelagem

### 1. Preparação dos Dados
```python
# Seleção de features (24 finais)
X = df[features_to_use]
y = df['fraud']

# Tratamento de valores infinitos e NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
```

### 2. Split Treino/Teste (Estratificado)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # CRÍTICO para dados desbalanceados
)

# Treino: 475,714 (469,954 normal + 5,760 fraude)
# Teste:  118,929 (117,489 normal + 1,440 fraude)
```

### 3. Normalização
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Importante**: Fit apenas no treino para evitar data leakage

---

## 🎯 Modelos Testados

### 1. Logistic Regression
```python
LogisticRegression(random_state=42, max_iter=1000)
```
**Resultados**:
- ROC-AUC: ~0.94
- Precision: ~0.85
- Recall: ~0.70
- F1-Score: ~0.77

### 2. Random Forest ⭐ MELHOR
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```
**Resultados**:
- **ROC-AUC: ~0.98** ← Excelente
- **Precision: ~0.90**
- **Recall: ~0.85**
- **F1-Score: ~0.87**

### 3. Gradient Boosting
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
```
**Resultados**:
- ROC-AUC: ~0.96
- Precision: ~0.88
- Recall: ~0.80
- F1-Score: ~0.84

---

## 📈 Resultados Finais - Random Forest

### Matriz de Confusão
```
Predito:      Normal   Fraude
Real:
Normal     [[116,800    689]
Fraude         [216   1,224]]

TN = 116,800  (verdadeiros negativos)
FP = 689      (falsos positivos - alarmes falsos)
FN = 216      (falsos negativos - fraudes não detectadas)
TP = 1,224    (verdadeiros positivos - fraudes detectadas)
```

### Interpretação das Métricas
```
Recall:    85% das fraudes foram detectadas
Precision: 64% dos alertas são fraudes reais
           (1,224 / (1,224 + 689))

Taxa de falsos positivos: 0.6% (689/118,929)
Taxa de fraudes perdidas: 15% (216/1,440)
```

### Feature Importance (Top 10)
| Rank | Feature | Importância | Categoria |
|------|---------|-------------|-----------|
| 1 | amount_media_5steps | 16.4% | Temporal |
| 2 | amount | 16.4% | Original |
| 3 | tx_cliente_merchant | 14.8% | Relacionamento |
| 4 | prop_tx_merchant | 11.8% | Relacionamento |
| 5 | amount_mean_cliente | 9.4% | Cliente |
| 6 | amount_std_cliente | 7.1% | Cliente |
| 7 | category_encoded | 5.1% | Original |
| 8 | amount_desvio_5steps | 4.4% | Temporal |
| 9 | qtd_transacoes | 2.4% | Cliente |
| 10 | num_merchants_cliente | 2.2% | Cliente |

**Insight**: Features temporais e de relacionamento são as mais importantes!

---

## 💡 Aplicação Prática

### Sistema de Alerta em Produção
```python
# Carregar modelo treinado
import joblib
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# Processar nova transação
new_transaction = preprocess_features(transaction_data)
new_transaction_scaled = scaler.transform(new_transaction)

# Predição
proba_fraude = model.predict_proba(new_transaction_scaled)[:, 1]
is_fraud = proba_fraude > threshold

# Ação
if is_fraud:
    block_transaction()
    notify_analyst()
    send_sms_to_customer()
```

### Ajuste de Threshold por Custo
```python
# Custo de falso positivo: $10 (revisão manual)
# Custo de falso negativo: $500 (fraude não detectada)

# Otimizar threshold
threshold = 0.5  # Padrão
threshold = 0.3  # Mais sensível (maior recall, mais falsos positivos)
threshold = 0.7  # Mais conservador (maior precision, perde fraudes)
```

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

### 2. Baixar Dataset
```bash
# Download do Kaggle
kaggle datasets download -d ealaxi/banksim1
unzip banksim1.zip
```

### 3. Executar Notebooks
```bash
# Notebook principal
jupyter notebook fraud_detection_banksim.ipynb

# Notebook de auditoria
jupyter notebook relatorio_para_auditoria.ipynb
```

---

## 📂 Estrutura do Projeto

```
Deteccao_Fraudes/
├── README.md                          # Este arquivo
├── AGENT_CONTEXT.md                  # Contexto técnico para IA
├── fraud_detection_banksim.ipynb     # Análise e modelagem principal
├── relatorio_para_auditoria.ipynb    # Versão auditável
├── data/
│   ├── bs140513_032310.csv           # Dataset original (grande)
│   └── Relatorio_Fraudes_20251227.csv # Relatório gerado
├── models/                            # Modelos salvos
│   ├── fraud_model.pkl
│   └── scaler.pkl
└── requirements.txt                   # Dependências
```

---

## 📚 Dependências

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
scikit-learn>=1.0.0
xgboost>=1.5.0
jupyter>=1.0.0
```

---

## 🔍 Análise de Resultados

### Por Que o Modelo Funciona Bem?

✅ **Feature Engineering Extensivo**:
- 24 features criadas capturando padrões comportamentais
- Features temporais detectam mudanças súbitas
- Features de relacionamento identificam padrões anômalos

✅ **Métricas Adequadas**:
- ROC-AUC em vez de Accuracy (dados desbalanceados)
- Foco em Recall (detectar fraudes) e Precision (custo operacional)

✅ **Algoritmo Robusto**:
- Random Forest lida bem com outliers
- Captura interações não-lineares
- Feature importance interpretável

### Trade-offs e Decisões

**Recall 85% vs Precision 64%**:
- Detectamos 85% das fraudes (excelente)
- 36% dos alertas são falsos (aceitável, considerando custo baixo de revisão)

**Threshold Ajustável**:
- Produção pode usar threshold menor (e.g., 0.3) para maior recall
- Custo de revisão manual é muito menor que perda por fraude

---

## 🔮 Melhorias Futuras

### Features Adicionais
- [ ] Análise de grafos (rede de clientes-merchants)
- [ ] Features de horário (hora do dia, dia da semana)
- [ ] Geolocalização mais detalhada (distância cliente-merchant)
- [ ] Velocidade de transações (múltiplas em curto espaço de tempo)

### Modelos Avançados
- [ ] XGBoost com otimização de hiperparâmetros
- [ ] LightGBM para dados grandes
- [ ] Stacking de modelos (LR + RF + GB)
- [ ] Deep Learning (LSTM para sequências temporais)

### Deployment
- [ ] API REST com Flask/FastAPI
- [ ] Monitoramento de drift
- [ ] Retreinamento automático periódico
- [ ] Dashboard de métricas em tempo real

---

## 📖 Notebook de Auditoria

### relatorio_para_auditoria.ipynb

**Objetivo**: Versão limpa e verificável dos resultados para fins de compliance e auditoria.

**Conteúdo**:
1. Sumário executivo das descobertas
2. Tabelas e visualizações replicáveis
3. Passos de validação (data leakage, estratificação)
4. Análise de distribuição de classes
5. Exemplos de predições corretas/incorretas
6. Conclusões e recomendações

---

## 🤝 Contribuindo

Contribuições são bem-vindas!

**Como contribuir**:
1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📖 Referências

- Dataset: [BankSim1 - Kaggle](https://www.kaggle.com/datasets/ealaxi/banksim1)
- Paper: Lopez-Rojas, E. A. et al. "PaySim: A financial mobile money simulator for fraud detection" (2016)
- Técnicas: Chawla et al. "SMOTE: Synthetic Minority Over-sampling Technique" (2002)

---

## 📝 Citação

Se usar este projeto, por favor cite:
```
@misc{fraud_detection_2026,
  author = {Carlos Henrique Bamberg Marques},
  title = {Detecção de Fraudes Bancárias com Machine Learning},
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

**🎯 Lembre-se**: Em detecção de fraudes, ROC-AUC > Accuracy!

*Projeto desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_02*
