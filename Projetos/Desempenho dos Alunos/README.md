
# 🎓 Análise de Desempenho de Alunos

Este projeto tem como objetivo realizar uma análise exploratória e modelagem preditiva do desempenho escolar de estudantes, com base em características demográficas, comportamentais e acadêmicas. O projeto integra o Módulo de **Machine Learning** da Especialização em IA.

---

## 📌 Objetivos

- Compreender os principais fatores que impactam o desempenho dos alunos.
- Explorar relações entre variáveis como faltas, notas, apoio familiar e consumo de álcool.
- Construir modelos de classificação do desempenho estudantil (multiclasse).
- Avaliar a viabilidade de prever o desempenho com os dados disponíveis.

---

## 🧭 Pipeline do Projeto

1. **Importação e Visualização Inicial**
2. **Análise Exploratória de Dados (EDA)**
   - Distribuição das variáveis
   - Correlações
   - Cruzamento entre notas e variáveis demográficas
3. **Pré-processamento**
   - Remoção de outliers (ex: faltas > 30)
   - Tratamento de dados faltantes
   - Encoding de variáveis categóricas (One-Hot)
   - Escalonamento de variáveis numéricas
4. **Modelagem**
   - Random Forest padrão
   - Random Forest com `class_weight='balanced'`
   - LightGBM
   - SMOTE para balanceamento de classes
   - Otimização de hiperparâmetros com RandomizedSearchCV
   - Reclassificação do alvo em 3 classes: Baixo, Médio, Alto
5. **Avaliação**
   - Classification Report
   - Métricas de acurácia, precisão, recall e f1-score

---

## 📉 Principais Resultados

Apesar dos esforços em balanceamento e otimização de modelos, a **acurácia permaneceu em torno de 50%**, mesmo após:

- Aplicação do SMOTE
- Reclassificação do target em 3 categorias
- Ajustes de hiperparâmetros

---

## ✅ Conclusões

- O dataset possui **valor para análise exploratória**, mas **não apresenta variáveis suficientes para uma predição confiável**.
- Os padrões encontrados podem **orientar ações educacionais**, como foco em alunos com alto número de faltas ou baixo suporte familiar.
- A modelagem foi útil para demonstrar **os limites do aprendizado supervisionado em bases restritas**.

---

## 📂 Estrutura do Projeto

Alunos/
├── student_datas/
│ ├── student-mat.csv
│ └── student-por.csv
├── notebooks/
│ └── alunos.ipynb
├── student.txt
├── README.md
└── requirements.txt


---

## 🛠️ Tecnologias Utilizadas

- Python 3.10+
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- LightGBM
- Imbalanced-learn (SMOTE)

---

## Instalar dependências

pip install -r requirements.txt

---

## Licença
Este projeto está licenciado sob a MIT License.
Veja o arquivo LICENSE para mais detalhes.

---

## ✍️ Autor

**Carlos Henrique Bamberg Marques**  
Email: rick.bamberg@gmail.com
GitHub: https://github.com/RickBamberg/
Desenvolvedor e entusiasta em Inteligência Artificial  
Especialização em IA com foco em portfólio e projetos aplicados.

---

