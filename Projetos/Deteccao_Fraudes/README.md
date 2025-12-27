# Detecção de Fraudes — Projeto

Projeto para análise e detecção de fraudes em dados bancários. Contém notebooks para análise contextual, modelagem de Machine Learning e um notebook focado em auditoria.

-**Estrutura do Repositório**
- **bs140513_032310.csv**: Conjunto de dados original (BankSim) utilizado para análise e treino. Fonte: https://www.kaggle.com/datasets/ealaxi/banksim1 — arquivo grande, pode não ser incluído diretamente no repositório por ser volumoso; baixe a partir do Kaggle se necessário.
- **Relatorio_Fraudes_20251227.csv**: Relatório de fraudes gerado/externo.
- **fraud_detection_banksim.ipynb**: Notebook com análise exploratória, pré-processamento e modelos de ML para detecção de fraudes.
- **relatorio_para_auditoria.ipynb**: Notebook com passos e outputs organizados para fins de auditoria (inclui relatórios e validações).

**Notebooks**
- **fraud_detection_banksim.ipynb**: Execução principal — contém:
  - Carregamento e limpeza dos dados.
  - Análise exploratória e visualizações.
  - Engenharia de features e balanceamento de classes.
  - Treinamento e avaliação de modelos (ex.: Logistic Regression, Random Forest, XGBoost).
  - Exportação de resultados e métricas.

- **relatorio_para_auditoria.ipynb**: Versão auditável dos resultados — contém:
  - Sumário executivo das descobertas.
  - Tabelas e visualizações replicáveis.
  - Passos e verificações para auditoria dos dados e do processo de modelagem.

**Requisitos (sugestão)**
Recomenda-se Python 3.8+ e criação de ambiente virtual. Bibliotecas comuns usadas:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
xgboost
jupyter
notebook
ipykernel

# (adicione outras conforme necessário)
```

**Instalação e execução rápida**
1. Criar e ativar ambiente virtual:

```bash
python -m venv .venv
.
source .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
```

2. Instalar dependências (ou criar `requirements.txt`):

```bash
pip install -r requirements.txt
# ou instalar as principais diretamente
pip install pandas numpy scikit-learn matplotlib seaborn plotly xgboost jupyter
```

3. Executar Jupyter Notebook / Jupyter Lab:

```bash
jupyter notebook
# ou
jupyter lab
```

4. Abrir e executar os notebooks:
- Abra `[fraud_detection_banksim.ipynb](fraud_detection_banksim.ipynb)` para análise e modelagem.
- Abra `[relatorio_para_auditoria.ipynb](relatorio_para_auditoria.ipynb)` para ver o relatório preparado para auditoria.

**Dados**
- Os arquivos CSV estão na raiz do repositório: `[bs140513_032310.csv](bs140513_032310.csv)` e `[Relatorio_Fraudes_20251227.csv](Relatorio_Fraudes_20251227.csv)`.

**Boas práticas**
- Versione o `requirements.txt` e notebooks limpos (remova saídas) antes de compartilhar.
- Documente qualquer transformação de dados crítica dentro dos notebooks para auditoria.

