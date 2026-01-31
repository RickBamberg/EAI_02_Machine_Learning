# Predição de Diabetes com Flask 🏥

Sistema completo de Machine Learning para predição de risco de diabetes, com aplicação web Flask deployada e testes automatizados.

---

## 📌 Sobre o Projeto

Este projeto implementa um **sistema end-to-end** de predição de diabetes, desde análise exploratória e modelagem até deployment de aplicação web com validação fisiológica e sistema de logging robusto.

**Diferencial**: Não é apenas um modelo, mas uma **aplicação completa** pronta para uso, com interface web, validação de inputs, testes automatizados e até geração de executável standalone.

---

## 🎯 Problema

**Tipo**: Classificação Binária  
**Target**: 0 = Não diabético | 1 = Diabético  
**Dataset**: Pima Indians Diabetes Database (768 registros)

**Objetivo**: Prever risco de diabetes baseado em 8 variáveis clínicas (gravidez, glicose, pressão arterial, IMC, idade, etc.) e disponibilizar via interface web.

---

## 📊 Dataset - Pima Indians

### Fonte
- **Nome**: Pima Indians Diabetes Database
- **URL**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **População**: Mulheres Pima de pelo menos 21 anos
- **Tamanho**: 768 registros

### Características
```
Total:            768 registros
Não diabético:    ~500 (65%)
Diabético:        ~268 (35%)
Features:         8 variáveis clínicas
```

### Estrutura do Dataset (9 colunas)

| Coluna Original | Nome PT | Tipo | Range Fisiológico | Descrição |
|----------------|---------|------|-------------------|-----------|
| Pregnancies | Gravidez | int | 0-20 | Número de gestações |
| Glucose | Glicose | int | 50-300 | Concentração de glicose (mg/dL) |
| BloodPressure | Pressão arterial | int | 40-140 | Pressão diastólica (mm Hg) |
| SkinThickness | Espessura da pele | int | 5-50 | Dobra cutânea tríceps (mm) |
| Insulin | Insulina | int | 0-1000 | Insulina sérica (μU/mL) |
| BMI | IMC | float | 15-50 | Índice de massa corporal |
| DiabetesPedigreeFunction | Diabetes Descendente | float | 0-1 | Função pedigree (histórico familiar) |
| Age | Idade | int | 15-100 | Idade (anos) |
| **Outcome** | **Resultado** | **int** | **0, 1** | **Target** |

### Problema: Zeros Implausíveis

**Colunas com zeros que não fazem sentido**:
- Glicose = 0 → impossível (pessoa estaria morta)
- Pressão arterial = 0 → impossível
- Espessura da pele = 0 → improvável
- IMC = 0 → impossível

**Solução**:
```python
# Substituir zeros por NaN
colunas_zero_invalido = ['Glicose', 'Pressão arterial', 'Espessura da pele', 'Insulina', 'IMC']
df[colunas_zero_invalido] = df[colunas_zero_invalido].replace(0, np.nan)

# Imputar pela mediana
df[colunas_zero_invalido] = df[colunas_zero_invalido].fillna(df[colunas_zero_invalido].median())
```

---

## 🔧 Pipeline de Machine Learning

### 1. Pré-processamento **Misto** (Diferencial do Projeto)

**Estratégia**: Aplicar scalers diferentes conforme características das features

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('robust', RobustScaler(), ['Insulina', 'Espessura da pele', 'IMC']),
        ('standard', StandardScaler(), ['Glicose', 'Pressão arterial']),
        ('pass', 'passthrough', ['Gravidez', 'Idade', 'Diabetes Descendente'])
    ],
    remainder='passthrough'
)
```

**Justificativa**:
- **RobustScaler**: Para features com outliers (Insulina tem valores muito dispersos)
- **StandardScaler**: Para features mais normais (Glicose tem distribuição mais regular)
- **Passthrough**: Features já em escala adequada (contagens, proporções)

**Resultado**: Pré-processamento misto gerou **+3-5% accuracy** vs scaler único!

### 2. Balanceamento com SMOTE
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    preprocessor.fit_transform(X_train), 
    y_train
)
```

**Importante**: SMOTE aplicado **apenas no treino** (evitar data leakage)

### 3. Modelagem - RandomForestClassifier
```python
model = RandomForestClassifier(
    n_estimators=200,  # 200 árvores
    max_depth=10,      # Profundidade limitada (evita overfit)
    random_state=42
)

model.fit(X_train_balanced, y_train_balanced)
```

---

## 📈 Resultados do Modelo

### Performance
```
Accuracy:   77-80%
Precision:  75-78%
Recall:     70-75%
F1-Score:   72-76%
```

### Feature Importance
| Rank | Feature | Importância | Interpretação |
|------|---------|-------------|---------------|
| 1 | Glicose | ~25% | Principal indicador de diabetes |
| 2 | IMC | ~18% | Fator de risco importante |
| 3 | Idade | ~15% | Risco aumenta com idade |
| 4 | Diabetes Descendente | ~12% | Genética é relevante |
| 5 | Gravidez | ~10% | Diabetes gestacional |
| 6 | Insulina | ~8% | Resistência insulínica |
| 7 | Pressão arterial | ~7% | Comorbidade |
| 8 | Espessura da pele | ~5% | Indicador indireto |

### Matriz de Confusão Típica
```
              Predito
              Não   Sim
Real  Não  [[95    15]
      Sim   [20    50]]

Accuracy: 79.3% ((95+50)/180)
Recall diabéticos: 71.4% (50/70)
Precision diabéticos: 76.9% (50/65)
```

---

## 🌐 Aplicação Web Flask

### Funcionalidades

✅ **Interface Web Completa**:
- Formulário para entrada de dados clínicos
- Validação fisiológica em tempo real
- Exibição clara do resultado e probabilidade

✅ **Validação Fisiológica**:
```python
RANGES = {
    'Gravidez': (0, 20),
    'Glicose': (50, 300),
    'Pressão arterial': (40, 140),
    'Espessura da pele': (5, 50),
    'Insulina': (0, 1000),
    'IMC': (15, 50),
    'Diabetes Descendente': (0, 1),
    'Idade': (15, 100)
}
```

✅ **Sistema de Logging**:
```python
logs/
├── app.log       # Logs gerais da aplicação
└── access.log    # Logs de requisições HTTP
```

✅ **Testes Automatizados** (`/test_model`):
- Carrega casos de teste de `data/test_cases.csv`
- Executa predições automaticamente
- Exibe tabela comparativa com acertos/erros

✅ **Executável Standalone**:
- Geração de `.exe` com PyInstaller
- Funciona sem Python instalado

---

## 🚀 Como Executar

### Opção 1: Executar Localmente (Recomendado)

#### 1. Configurar Ambiente
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Linux/macOS)
source venv/bin/activate

# Ativar (Windows)
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

#### 2. Treinar e Salvar Modelo
```bash
python salva_modelo.py
```

**Output**:
```
✅ Modelo e pré-processador salvos com sucesso!
model/model.pkl
model/preprocessor.pkl
```

#### 3. Rodar Aplicação Flask
```bash
python app.py
```

**Acesse**: http://localhost:5000

#### 4. Testar com Casos Pré-definidos
**Acesse**: http://localhost:5000/test_model

---

### Opção 2: Executável Windows (.exe)

#### 1. Instalar PyInstaller
```bash
pip install pyinstaller
```

#### 2. Gerar Executável
```bash
# Windows
gerar_executavel.bat

# Ou manualmente
pyinstaller --onefile --windowed ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --add-data "model;model" ^
  --add-data "data;data" ^
  app.py
```

#### 3. Executar
```bash
dist/app.exe
```

**Acesse**: http://localhost:5000

---

## 📂 Estrutura do Projeto

```
Diabetes/
├── README.md                    # Este arquivo
├── AGENT_CONTEXT.md            # Contexto técnico para IA
├── Diabetes.ipynb              # Notebook de análise e modelagem
│
├── app.py                      # ⭐ Aplicação Flask principal
├── salva_modelo.py             # Script de treino
├── gerar_executavel.bat        # Gera .exe
│
├── model/
│   ├── model.pkl               # Modelo treinado
│   └── preprocessor.pkl        # Preprocessador
│
├── data/
│   ├── diabetes.csv            # Dataset original
│   ├── diabetesX.csv           # Dataset processado
│   └── test_cases.csv          # Casos de teste
│
├── templates/
│   ├── index.html              # Formulário de entrada
│   ├── results.html            # Resultado da predição
│   └── test_results.html       # Testes automatizados
│
├── static/
│   └── style.css               # Estilos CSS
│
├── util/
│   └── logging_config.py       # Configuração de logging
│
├── logs/
│   ├── app.log                 # Logs da aplicação
│   └── access.log              # Logs de acesso
│
└── requirements.txt            # Dependências
```

---

## 💻 Código-Chave

### app.py - Rota Principal
```python
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Mapeamento form → modelo
        features = {
            'Gravidez': float(request.form['gravidez']),
            'Glicose': float(request.form['glicose']),
            'Pressão arterial': float(request.form['pressao']),
            'Espessura da pele': float(request.form['pele']),
            'Insulina': float(request.form['insulina']),
            'IMC': float(request.form['IMC']),
            'Diabetes Descendente': float(request.form['historia']),
            'Idade': float(request.form['idade'])
        }
        
        # Validação fisiológica
        if errors := validate_input(features):
            return render_template('index.html', errors=errors)
        
        # Predição
        df = pd.DataFrame([features])
        X = preprocessor.transform(df)
        
        proba = model.predict_proba(X)[0]
        resultado = model.predict(X)[0]
        
        return render_template('results.html',
                           resultado=resultado,
                           probabilidade=proba[1]*100,
                           features=features)
    
    return render_template('index.html')
```

### salva_modelo.py - Treino
```python
# 1. Carregar dados
df = pd.read_csv('data/diabetesX.csv')
X_train = df.drop('Resultado', axis=1)
y_train = df['Resultado']

# 2. Definir preprocessador
preprocessor = ColumnTransformer([
    ('robust', RobustScaler(), ['Insulina', 'Espessura da pele', 'IMC']),
    ('standard', StandardScaler(), ['Glicose', 'Pressão arterial']),
    ('pass', 'passthrough', ['Gravidez', 'Idade', 'Diabetes Descendente'])
])

# 3. Treinar
X_train_processed = preprocessor.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_processed, y_train)

# 4. Salvar
joblib.dump(preprocessor, 'model/preprocessor.pkl')
joblib.dump(model, 'model/model.pkl')
```

---

## 🧪 Testes Automatizados

### test_cases.csv
```csv
Gravidez,Glicose,Pressão arterial,Espessura da pele,Insulina,IMC,Diabetes Descendente,Idade,Resultado_Esperado
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
```

### Rota /test_model
1. Carrega `test_cases.csv`
2. Para cada caso:
   - Faz predição
   - Compara com resultado esperado
   - Calcula probabilidade
3. Exibe tabela HTML com acertos/erros

**Exemplo de Output**:
```
Caso 1: ✓ Correto (Previsto: 1, Esperado: 1, Prob: 87%)
Caso 2: ✓ Correto (Previsto: 0, Esperado: 0, Prob: 23%)
Caso 3: ✗ Errado  (Previsto: 0, Esperado: 1, Prob: 45%)
```

---

## 📚 Dependências

### requirements.txt
```txt
flask>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.9.0
jupyter>=1.0.0
```

---

## 🎯 Comparação de Estratégias

### Pré-processamento Testado

| Estratégia | Accuracy | Observação |
|-----------|----------|------------|
| StandardScaler (tudo) | ~75% | Baseline |
| MinMaxScaler (tudo) | ~74% | Não ajudou |
| RobustScaler (tudo) | ~76% | Melhor que Standard |
| **Misto (ColumnTransformer)** | **~77-80%** | ⭐ **Melhor** |

**Conclusão**: Pré-processamento diferenciado por tipo de feature melhora performance!

---

## 💡 Validação Fisiológica - Por Que?

### Benefícios

✅ **Evita Inputs Impossíveis**:
- Glicose 0 → pessoa estaria morta
- Idade 200 → impossível
- IMC 5 → desnutrição extrema

✅ **Melhora UX**:
- Feedback imediato ao usuário
- Mensagens de erro claras

✅ **Previne Erros de Modelo**:
- Valores fora do range de treino podem gerar predições ruins

### Ranges Definidos
```python
RANGES = {
    'Gravidez': (0, 20),           # Máximo ~17 registrado
    'Glicose': (50, 300),          # <50 hipoglicemia, >300 cetoacidose
    'Pressão arterial': (40, 140), # 40 choque, 140 hipertensão
    'Espessura da pele': (5, 50),  # 5mm mínimo razoável
    'Insulina': (0, 1000),         # 0 jejum válido, >1000 raro
    'IMC': (15, 50),               # <15 desnutrição, >50 obesidade mórbida
    'Diabetes Descendente': (0, 1), # Normalizado
    'Idade': (15, 100)             # Dataset: 21+, aceitamos 15+
}
```

---

## 🔮 Melhorias Futuras

### Modelo
- [ ] Otimização de hiperparâmetros (GridSearchCV)
- [ ] Ensemble de modelos (RF + XGBoost + LightGBM)
- [ ] Feature engineering (interações: Glicose × IMC)
- [ ] Threshold de probabilidade ajustável

### Aplicação
- [ ] Deploy em cloud (Heroku, Railway, AWS)
- [ ] Docker container
- [ ] API REST para integração
- [ ] Dashboard de predições
- [ ] Autenticação de usuários

### Interface
- [ ] Máscaras de input mais amigáveis
- [ ] Validação client-side (JavaScript)
- [ ] Gráficos de probabilidade
- [ ] Histórico de predições

### Testes
- [ ] Testes unitários (pytest)
- [ ] Testes de integração
- [ ] CI/CD (GitHub Actions)

---

## 🤝 Contribuindo

Contribuições são bem-vindas!

**Como contribuir**:
1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

---

## 📖 Referências

- Dataset: [Pima Indians Diabetes - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Paper Original: Smith et al. "Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus" (1988)
- Flask: [Flask Documentation](https://flask.palletsprojects.com/)
- SMOTE: Chawla et al. "SMOTE: Synthetic Minority Over-sampling Technique" (2002)

---

## 📝 Citação

Se usar este projeto, por favor cite:
```
@misc{diabetes_predictor_2026,
  author = {Carlos Henrique Bamberg Marques},
  title = {Predição de Diabetes com Flask - Sistema End-to-End},
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

**🎯 Diferencial**: Não é só um modelo, é uma aplicação completa pronta para uso!

*Projeto desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_02*
