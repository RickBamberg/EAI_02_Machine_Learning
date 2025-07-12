import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler
import logging # Import logging
import os 

# Importe e execute a configuração de logging
from util.logging_config import setup_loggers
setup_loggers()

# Obtenha o logger configurado (com o mesmo nome usado na configuração)
logger = logging.getLogger('app') # Usando 'app' para que vá para app.log e console

# 1. Carregar seus dados originais
try:
    logger.info("Iniciando script salva_modelo.py")
    logger.info("Carregando dados originais de 'data/diabetesX.csv'") # Ajuste o caminho/nome
    df = pd.read_csv('data/diabetesX.csv') # Use 'data' em vez de 'date'?
    logger.debug(f"Colunas carregadas: {df.columns.tolist()}")

    # Verificar se a coluna target existe
    target_column = 'Resultado'
    if target_column not in df.columns:
        # Use logger.error ou logger.critical aqui
        logger.critical(f"Coluna target '{target_column}' não encontrada. Colunas disponíveis: {df.columns.tolist()}")
        raise ValueError(f"Coluna target '{target_column}' não encontrada.")

    X_train = df.drop(target_column, axis=1)
    y_train = df[target_column]
    logger.info(f"Dados carregados: {X_train.shape[0]} linhas.")

    # 2. Definir o pré-processador
    logger.info("Definindo o pré-processador ColumnTransformer")
    numeric_features = ['Insulina', 'Espessura da pele', 'IMC', 'Glicose', 'Pressão arterial']
    pass_features = ['Gravidez', 'Idade', 'Diabetes Descendente']

    # Verificar se todas as colunas existem
    missing_cols = [col for col in numeric_features + pass_features if col not in X_train.columns]
    if missing_cols:
        logger.critical(f"Colunas do pré-processador não encontradas nos dados: {missing_cols}.")
        raise ValueError(f"Colunas não encontradas: {missing_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('robust', RobustScaler(), ['Insulina', 'Espessura da pele', 'IMC']),
            ('standard', StandardScaler(), ['Glicose', 'Pressão arterial']),
            ('pass', 'passthrough', pass_features)
        ],
        remainder='passthrough'
    )

    # 3. Treinar o pipeline
    logger.info("Ajustando (fit) o pré-processador e transformando os dados de treino...")
    X_train_processed = preprocessor.fit_transform(X_train)
    logger.info("Dados de treino pré-processados.")

    logger.info("Treinando o modelo RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train_processed, y_train)
    logger.info("Modelo treinado com sucesso.")

    # 4. Salvar os artefatos
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"Diretório '{model_dir}' criado.")

    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    model_path = os.path.join(model_dir, 'model.pkl')

    logger.info(f"Salvando pré-processador em '{preprocessor_path}'")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Salvando modelo em '{model_path}'")
    joblib.dump(model, model_path)

    logger.info("✅ Modelo e pré-processador salvos com sucesso!")

except FileNotFoundError as e:
    logger.critical(f"Erro: Arquivo de dados não encontrado. Verifique o caminho/nome. Detalhes: {e}", exc_info=True)
except ValueError as e:
    logger.critical(f"Erro de valor (ex: coluna faltando): {e}", exc_info=True)
except Exception as e:
    logger.critical(f"❌ Ocorreu um erro inesperado durante o salvamento: {e}", exc_info=True)
    logger.critical("Verifique os logs e a configuração.")