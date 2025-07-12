# Predição de Diabetes com Flask e Machine Learning

Projeto para predição de risco de diabetes a partir de dados clínicos usando modelo RandomForest treinado com dados públicos, integrado a uma aplicação web feita em Flask.

---

## Sobre o Projeto

Este projeto realiza a predição do risco de diabetes com base em dados informados pelo usuário via formulário web.

O modelo foi treinado usando RandomForestClassifier com pré-processamento aplicado nos dados clínicos originais.

A aplicação Flask carrega o modelo e o pré-processador, valida as entradas do usuário, realiza a predição e exibe o resultado de forma clara.

---

## Estrutura do Projeto

Diabetes/
├── app.py  # Aplicação Flask
├── salva_modelo.py  # Script para treinar e salvar o modelo e pré-processador
├── gerar_executavel.bat  # Gera programa executável
├── model/
│   ├── model.pkl  # Modelo treinado serializado
│   └── preprocessor.pkl  # Pré-processador serializado
├── data/
│   ├── diabetes.csv  # Base de dados usada para treino
│   └── test_cases.csv  # Lista de dadis para treino
├── templates/
│   ├── index.html  # Página inicial com formulário
│   ├── results.html  # Página com resultado da predição
│   └── test_results.html  # Página com resultados de testes automatizados
├── static/
│   └── style.css  # Estilos CSS para a aplicação web
├── util/
│   └── logging_config.py  # Configuração de logging customizado
├── requirements.txt   # Dependências do projeto
├── README.md  # Documentação do projeto (este arquivo)

└── logs/
    ├── app.log  # Arquivo de log do Aplicativo
    └── access.log  # Arquivo de log do Sistema

---

## Como usar

### 1. Configurar ambiente virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

### 2. Instalar dependências

pip install -r requirements.txt

### 3. Treinar e salvar o modelo
Execute o script que faz o treinamento do modelo e salva os artefatos:

python salva_modelo.py

Isso vai gerar os arquivos model/model.pkl e model/preprocessor.pkl.

### 4. Rodar a aplicação Flask

python app.py

Acesse a aplicação em: http://localhost:5000

### 5. Testar o modelo com casos pré-definidos

Abra no navegador:  # Teste com Casos Pré-definidos

http://localhost:5000/test_model  # para verificar os resultados de predição em um conjunto de testes localizado em data/test_cases.csv.

## Gerar Executável (.exe)
Este projeto pode ser convertido para um app desktop no Windows:

Pré-requisitos:

pip install pyinstaller

### Como gerar:
Execute o script .bat fornecido:

gerar_executavel.bat

O arquivo app.exe será gerado em dist/, e pode ser executado mesmo sem Python instalado.

## Sobre os dados 
 
Os dados usados para treinamento foram obtidos da base pública da "Pima Indians Diabetes Database", 
URL: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database.

## Logs

A aplicação grava logs detalhados na pasta logs/:

- app.log — logs gerais da aplicação

- access.log — logs das requisições HTTP (via Werkzeug)

## Melhorias Futuras  
- Validação e máscaras mais amigáveis no frontend

- Testes unitários para funções-chave

- Dockerização para facilitar deploy

- Suporte para deploy em plataformas cloud (Heroku, Railway, etc.)

- Documentação com exemplos de entrada e saída

## Licença
Este projeto está licenciado sob a MIT License.
Veja o arquivo LICENSE para mais detalhes.

## Contato
Carlos Henrique Bamberg Marques
Email: rick.bamberg@gmail.com
GitHub: https://github.com/RickBamberg/






