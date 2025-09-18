import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from datetime import datetime
from IPython.display import display

from sklearn.linear_model import (
    LogisticRegression, 
    RidgeClassifier, 
    SGDClassifier,
    PassiveAggressiveClassifier
)
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    ExtraTreesClassifier, 
    HistGradientBoostingClassifier
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, 
    QuadraticDiscriminantAnalysis
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV

from sklearn.model_selection import (
    StratifiedKFold, 
    train_test_split,
)

from sklearn.metrics import (
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)

from datetime import datetime
now = datetime.now()
experiment_name_path = 'experiment_17_09_2025' #f'experiment_{now.strftime("%d_%m_%Y")}'

# mapeando o diretório do projeto e do arquivo notebook 
diretorio_atual_projeto = os.getcwd() # Diretório atual do arquivo
notebook_dir_project_predict = os.path.normpath(f"{diretorio_atual_projeto}{os.sep}..{os.sep}..{os.sep}") + os.sep # Diretório do projeto
print(diretorio_atual_projeto)
print(notebook_dir_project_predict)

output_dir = f'{notebook_dir_project_predict}data{os.sep}output{os.sep}2023{os.sep}output_{experiment_name_path}{os.sep}test_train_path'

os.makedirs(output_dir, exist_ok=True)


df_limpo_final = pd.read_csv(f'{notebook_dir_project_predict}data{os.sep}output{os.sep}2023{os.sep}output_{experiment_name_path}{os.sep}brfss_2023_cleaned_to_model.csv')

### Retirada das Colunas
df_limpo_final = df_limpo_final.drop(columns=[ 'CVDINFR4', 'CVDCRHD4', '_STATE'])
# as colunas retiradas acima foram usadas para produzir a _MICHD 
# assim retiramos para efeito de trinamento e teste 

### Definindo o nome da coluna alvo
NOME_COLUNA_ALVO = '_MICHD'

### Recategorização da coluna alvo

# 0 são os casos NEGATIVOS 
# 1 são os casos POSITIVOS

df_limpo_final[NOME_COLUNA_ALVO] = df_limpo_final[NOME_COLUNA_ALVO].replace({2.0: 0, 1.0: 1})

# lista de colunas do dataframe que serão consultadas no codebook
list_columns_df_to_model = df_limpo_final.columns.tolist()

# codebook 
codebook_file_final =  pd.read_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}intermediate_{experiment_name_path}{os.sep}brfss_2023_variaveis_expandidas_translated_categor_continu.csv")

# traga de codebook_file_final os registros onde a var sas name está em list_columns_df_to_model
codebook_filtered = codebook_file_final[codebook_file_final['SAS Variable Name'].isin(list_columns_df_to_model)]

# retorne a primeira ocorrencia de cada variável sas name
codebook_filtered = codebook_filtered.drop_duplicates(subset=['SAS Variable Name'], keep='first')

# retirar daqui  _MICHD da codebook_filtered
codebook_filtered = codebook_filtered[~codebook_filtered['SAS Variable Name'].isin(['_MICHD', 'CVDINFR4', 'CVDCRHD4', '_STATE'])]

# podemos separar aqui as colunas que serão dummyizadas
# de codebook_filtered salve em duas listas de variavel sas name onde a coluna type of value é continuou ou categorical

list_sas_name_continuous = codebook_filtered[codebook_filtered['Type of Value'] == 'continuous']['SAS Variable Name'].tolist()
list_sas_name_categorical = codebook_filtered[codebook_filtered['Type of Value'] == 'categorical']['SAS Variable Name'].tolist()



# --- 2. Execução da Lógica ---

df = df_limpo_final.copy()
print(f"Dados carregados. Shape: {df.shape}")

# Separar X e y
X = df.drop(columns=[NOME_COLUNA_ALVO])
y = df[NOME_COLUNA_ALVO]

# Divisão Estratificada em Treino e Teste
print("\nDividindo os dados em 80% treino e 20% teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Divisão concluída. Shapes: X_train({X_train.shape}), X_test({X_test.shape})")

# --- One-Hot Encoding (Dummização) ---
print(f"\nAplicando One-Hot Encoding em {len(list_sas_name_categorical)} colunas categóricas...")
X_train = pd.get_dummies(X_train, columns=list_sas_name_categorical, dtype=int)
X_test = pd.get_dummies(X_test, columns=list_sas_name_categorical, dtype=int)

# Alinhar colunas para garantir que treino e teste tenham as mesmas features
# (isso lida com casos raros onde uma categoria só aparece em um dos conjuntos)
X_train_final, X_test_final = X_train.align(X_test, join='left', axis=1, fill_value=0)
print(f"Dummização concluída. Novo shape de X_train_final: {X_train_final.shape}")


# --- Normalização APENAS nas colunas quantitativas ---
print(f"\nAplicando StandardScaler em {len(list_sas_name_continuous)} colunas quantitativas...")
scaler = StandardScaler()

# Ajustar o scaler APENAS nos dados de treino
scaler.fit(X_train_final[list_sas_name_continuous])

# Aplicar a transformação nos dados de treino e teste
X_train_final[list_sas_name_continuous] = scaler.transform(X_train_final[list_sas_name_continuous])
X_test_final[list_sas_name_continuous] = scaler.transform(X_test_final[list_sas_name_continuous])

print("Normalização concluída.")

# --- Salvar os 4 arquivos ---
os.makedirs(output_dir, exist_ok=True)

X_train_final.to_csv(os.path.join(output_dir , 'X_train.csv'), index=False)
X_test_final.to_csv(os.path.join(output_dir , 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(output_dir , 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(output_dir , 'y_test.csv'), index=False)

print(f"\nArquivos de treino e teste, processados e prontos para modelagem, foram salvos na pasta: {output_dir}")



