# %% [markdown]
# # Libs

# %%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import seaborn as sns
# try lib polars
import polars as pl

# %%
'''  
Pré-processamento sugerido:

Remoção de valores ambíguos como "Don’t know", "Refused to answer".

Eliminação de colunas com mais de 30% de valores faltantes.

Substituição de códigos como 88/888 por zero em variáveis de contagem de dias.

Aplicação de normalização MinMaxScaler.

Redução de colinearidade com análise de correlação (Pearson).

Binning em variáveis contínuas como altura e peso.

'''

# %% [markdown]
# # Functions

# %%
# mapear o DataFrame de acordo com o codebook 

def mapear_colunas_para_rotulo(df_brfss, codebook_df, sufixo='_map'):
    """
    Substitui colunas do DataFrame por versões mapeadas com rótulos do codebook.
    A coluna original é excluída e substituída por uma nova com sufixo (default: _map).
    Se o valor não for mapeável, mantém o valor original.

    Parâmetros:
        df_brfss (pd.DataFrame): dados originais
        codebook_df (pd.DataFrame): DataFrame do codebook com 'SAS Variable Name', 'Value', 'Value Label'
        sufixo (str): sufixo para a nova coluna (default: '_map')

    Retorno:
        pd.DataFrame com colunas mapeadas
    """
    def limpar_valor(v):
        if pd.isna(v):
            return "BLANK"
        try:
            return str(int(v))  # converte 1.0 → '1'
        except:
            return str(v).strip()

    df_resultado = df_brfss.copy()
    colunas_mapeadas = {}

    with tqdm(total=len(df_brfss.columns), desc="Processing columns") as pbar:

        for coluna in df_brfss.columns:
            pbar.update(1)
            try:
                # Extrai mapeamentos apenas para a variável atual
                codebook_var = codebook_df[codebook_df['SAS Variable Name'] == coluna]
                codebook_var = codebook_var.dropna(subset=['Value'])

                if codebook_var.empty:
                    continue

                # Cria o dicionário de mapeamento
                mapa_valores = dict(zip(
                    codebook_var['Value'].astype(str).str.strip(),
                    codebook_var['Value Label']
                ))

                if not mapa_valores:
                    continue

                # Aplica mapeamento apenas onde existir valor no dicionário
                serie_convertida = df_brfss[coluna].apply(limpar_valor)
                serie_mapeada = serie_convertida.apply(lambda x: mapa_valores.get(x, x))  # mantém valor original se não estiver no dicionário
                colunas_mapeadas[coluna + sufixo] = serie_mapeada
            
            except Exception as e:
                print(f"Erro ao mapear coluna '{coluna}': {e}")
                continue

            

    # Cria DataFrame com as colunas mapeadas
    df_mapeadas = pd.DataFrame(colunas_mapeadas)

    # Remove colunas originais que foram mapeadas
    colunas_para_remover = [col.replace(sufixo, '') for col in tqdm(df_mapeadas.columns, desc='Remove suport columns')]
    df_resultado = df_resultado.drop(columns=colunas_para_remover)

    # Junta com colunas mapeadas
    df_resultado = pd.concat([df_mapeadas, df_resultado], axis=1)

    return df_resultado

# lista de colunas onde devemos retirar da string os seguintes caracteres "b'" di começo e "'" no final # IDATE IMONTH IDAY IYEAR
# IDATE IMONTH IDAY IYEAR
def limpar_colunas_data(df):
    """
    Função para limpar as colunas de data do DataFrame
    """
    # Limpando as colunas
    df['IDATE'] = df['IDATE'].str.replace("b'", "").str.replace("'", "")
    df['IMONTH'] = df['IMONTH'].str.replace("b'", "").str.replace("'", "")
    df['SEQNO'] = df['SEQNO'].str.replace("b'", "").str.replace("'", "")
    # df['IDAY'] = df['IDAY'].str.replace("b'", "").str.replace("'", "")
    # df['IYEAR'] = df['IYEAR'].str.replace("b'", "").str.replace("'", "")
    
    return df

# faremos uma lista de colunas que serão ignoradas pois todos os dados são o mesmo valor ou nulo e são dados de identificação de amostra





# %%
def read_data_and_codebook(path_csv_data, path_csv_codebook):
    """
    Função para ler os dados e o código  .
    """
    # Lendo os dados
    df = pd.read_csv(path_csv_data)
    
    # Lendo o código
    codebook = pd.read_csv(path_csv_codebook)
    
    return df, codebook

# transformar os dados categorizados como dont know ou refused em NaN
def transform_dont_know_refused_to_nan(df):
    """
    Função para transformar os dados categorizados como dont know ou refused em NaN
    """
    # Transformando os dados categorizados como dont know ou refused em NaN
    df = df.replace({'dont know': np.nan, 'refused': np.nan})
    
    return df

# retirar as colunas com mais de 30% de missing data conforme artigo
def remove_columns_with_missing_data(df, threshold=0.3):
    """
    Função para remover colunas com mais de 30% de missing data
    """
    # Calculando o percentual de missing data
    missing_data = df.isnull().mean()
    
    # Removendo as colunas com mais de 30% de missing data
    df = df.loc[:, missing_data < threshold]
    
    return df

# %%
# implementada a substituição por vazio 
def substituir_valores_por_zero_baseado_no_codebook(df_brfss, codebook_df, strings_para_zero, sufixo=''):
    """
    Processa colunas do DataFrame:
    Se o rótulo de um valor no codebook contiver alguma das 'strings_para_zero',
    o valor correspondente no DataFrame de dados é substituído por 0.
    Caso contrário, o valor original no DataFrame de dados é mantido.
    As colunas processadas substituem as originais com um sufixo.

    Parâmetros:
        df_brfss (pd.DataFrame): DataFrame original com os dados.
        codebook_df (pd.DataFrame): DataFrame do codebook com 'SAS Variable Name', 'Value', 'Value Label'.
        strings_para_zero (list): Lista de strings que, se encontradas no 'Value Label'
                                  do codebook, farão com que o valor original no df_brfss
                                  seja substituído por 0.
        sufixo (str): Sufixo para as novas colunas processadas.

    Retorno:
        pd.DataFrame com as colunas processadas.
    """

    def limpar_valor_para_lookup(v):
        """Limpa e converte valor para string para lookup no dicionário do codebook."""
        if pd.isna(v):
            return "INTERNAL_NAN_REPR" # Representação interna para NaNs originais dos dados
        try:
            # Tenta converter para int (para lidar com 1.0 -> '1'), depois para string
            return str(int(float(v)))
        except ValueError:
            # Se não puder ser convertido para float/int, usa como string
            return str(v).strip()
        except Exception:
            return str(v).strip() # Fallback

    df_processado = df_brfss.copy()
    colunas_originais_para_remover = []

    strings_para_zero_lower = [s.lower() for s in strings_para_zero]

    with tqdm(total=len(df_processado.columns), desc="Processando colunas") as pbar:
        for coluna in df_processado.columns:
            pbar.update(1)
            
            # Pega as entradas do codebook para a coluna atual
            codebook_var_atual = codebook_df[codebook_df['SAS Variable Name'] == coluna]
            
            if codebook_var_atual.empty:
                continue # Pula para a próxima coluna se não houver info no codebook

            # Cria um mapa de código (Value) para rótulo (Value Label)
            # Limpa os 'Value' do codebook para string para consistência
            mapa_codigo_rotulo = dict(zip(
                codebook_var_atual['Value'].apply(limpar_valor_para_lookup),
                codebook_var_atual['Value Label']
            ))
            
            # Série original da coluna a ser processada
            serie_original = df_processado[coluna].copy()
            # Série que será modificada (começa como uma cópia)
            serie_modificada = df_processado[coluna].copy()

            for idx, valor_original_na_serie in serie_original.items():
                valor_limpo_dados = limpar_valor_para_lookup(valor_original_na_serie)
                
                # Pega o rótulo do codebook para o valor limpo dos dados
                rotulo_do_codebook = mapa_codigo_rotulo.get(valor_limpo_dados)

                if rotulo_do_codebook: # Se encontrou um rótulo no codebook
                    # Verifica se alguma das strings_para_zero está no rótulo
                    if any(s_lower in str(rotulo_do_codebook).lower() for s_lower in strings_para_zero_lower):
                        serie_modificada.loc[idx] = np.nan # 0
                    # else: o valor original já está em serie_modificada, então não faz nada
                elif valor_limpo_dados == "INTERNAL_NAN_REPR" and "blank" in strings_para_zero_lower:
                    # Trata NaNs originais nos dados se "blank" for uma string para zerar
                    serie_modificada.loc[idx] = np.nan # 0
                # else: valor não encontrado no codebook ou rótulo não corresponde, mantém original

            # Atualiza a coluna no DataFrame processado
            df_processado[coluna + sufixo] = serie_modificada
            if sufixo : # Adiciona à lista para remover depois, apenas se houver sufixo
                colunas_originais_para_remover.append(coluna)
    
    # Remove as colunas originais que foram processadas (se o sufixo for diferente de vazio)
    if sufixo and colunas_originais_para_remover:
        colunas_existentes_para_remover = [col for col in colunas_originais_para_remover if col in df_processado.columns]
        df_processado.drop(columns=colunas_existentes_para_remover, inplace=True)
        
    return df_processado

# %%
# remoção de strings indesejadas

def limpar_valores_indesejados_codebook(df_brfss, codebook_df, rotulos_invalidos):
    """
    Substitui por NaN os valores do DataFrame que correspondem a rótulos inválidos do codebook.

    Parâmetros:
        df_brfss (pd.DataFrame): dados originais com valores numéricos
        codebook_df (pd.DataFrame): codebook com colunas 'SAS Variable Name', 'Value', 'Value Label'
        rotulos_invalidos (list): lista de strings com rótulos que devem ser tratados como NaN

    Retorno:
        pd.DataFrame com valores substituídos por NaN onde os rótulos são inválidos
    """
    df_resultado = df_brfss.copy()

    for coluna in df_resultado.columns:
        try:
            codebook_var = codebook_df[codebook_df['SAS Variable Name'] == coluna]
            codebook_var = codebook_var.dropna(subset=['Value', 'Value Label'])

            # Filtra os valores que têm rótulos indesejados
            valores_invalidos = codebook_var[
                codebook_var['Value Label'].str.strip().isin(rotulos_invalidos)
            ]['Value']

            # Converte para float para comparar com os dados
            valores_invalidos_float = valores_invalidos.astype(float).tolist()

            # Substitui no dado
            df_resultado[coluna] = df_resultado[coluna].apply(
                lambda x: np.nan if x in valores_invalidos_float else x
            )

        except Exception as e:
            print(f"Erro ao processar coluna '{coluna}': {e}")
            continue

    return df_resultado


# %%
# retirar as linhas onde a coluna MICHD for nulo

def remover_linhas_com_alvo_nulo(df, nome_coluna_alvo):
    """
    Remove linhas de um DataFrame onde a coluna alvo especificada é nula (NaN).

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        nome_coluna_alvo (str): Nome da coluna alvo para verificar valores nulos.

    Retorno:
        pd.DataFrame: DataFrame com as linhas nulas na coluna alvo removidas.
    """
    if nome_coluna_alvo not in df.columns:
        print(f"Erro: A coluna '{nome_coluna_alvo}' não existe no DataFrame.")
        return df # Retorna o DataFrame original se a coluna não existir

    linhas_antes = len(df)
    df_processado = df.dropna(subset=[nome_coluna_alvo])
    linhas_depois = len(df_processado)
    
    print(f"Coluna alvo para remoção de nulos: '{nome_coluna_alvo}'")
    print(f"Linhas antes da remoção: {linhas_antes}")
    print(f"Linhas removidas: {linhas_antes - linhas_depois}")
    print(f"Linhas após a remoção: {linhas_depois}")
    
    return df_processado

# %%
def contar_com_rotulo_args(
    df_brfss: pd.DataFrame,
    codebook_df: pd.DataFrame,
    coluna: str,
    ordenar_por: str = 'frequencia'  # ou 'valor'
) -> pd.Series:
    
    if coluna not in df_brfss.columns:
        print(f"Coluna '{coluna}' não encontrada no DataFrame BRFSS.")
        return pd.Series(dtype=int)

    codebook_var = codebook_df[codebook_df['SAS Variable Name'] == coluna]
    if codebook_var.empty:
        print(f"Variável '{coluna}' não encontrada no codebook.")
        return df_brfss[coluna].value_counts(dropna=False)

    # Remove linhas com 'Value' ausente
    codebook_var = codebook_var.dropna(subset=['Value'])

    # Mapeamento valor -> rótulo
    mapa_valores = dict(zip(
        codebook_var['Value'].astype(str).str.strip(),
        codebook_var['value_label_translate']
    ))

    # Converte os valores do dataframe para string (com tratamento de NaNs)
    def limpar_valor(v):
        if pd.isna(v):
            return "BLANK"
        try:
            return str(int(v))
        except:
            return str(v).strip()

    serie_convertida = df_brfss[coluna].apply(limpar_valor)
    serie_traduzida = serie_convertida.map(mapa_valores).fillna("Desconhecido")
    nome_variavel = codebook_var['Label'].dropna().iloc[0] if not codebook_var['Label'].dropna().empty else coluna
    serie_traduzida.name = nome_variavel

    contagem = serie_traduzida.value_counts(dropna=False)

    # Reordenar, se necessário
    if ordenar_por == 'valor':
        ordem_valores = codebook_var['Value'].astype(str).str.strip().tolist()
        ordem_legendas = [mapa_valores.get(val, 'Desconhecido') for val in ordem_valores]
        contagem = contagem.reindex(ordem_legendas).dropna()

    return contagem

# %% [markdown]
# # Vars

# %%
from datetime import datetime
now = datetime.now()
experiment_name_path = 'experiment_17_09_2025' #f'experiment_{now.strftime("%d_%m_%Y")}'

# mapeando o diretório do projeto e do arquivo notebook 
diretorio_atual_projeto = os.getcwd() # Diretório atual do arquivo
notebook_dir_project_predict = os.path.normpath(f"{diretorio_atual_projeto}{os.sep}..{os.sep}..{os.sep}") + os.sep # Diretório do projeto
print(diretorio_atual_projeto)
print(notebook_dir_project_predict)

csv_output_dir = f'{notebook_dir_project_predict}data{os.sep}output{os.sep}2023{os.sep}output_{experiment_name_path}'

os.makedirs(csv_output_dir, exist_ok=True)



# %% [markdown]
# # Scripts

# %% [markdown]
# Ler o codebook e o respectivo dado

# %%
'''       
No codebook,
SAS Variable Name - Coluna do Dado bruto 
Value - Valor no dado bruto
Value Label - Descrição do valor no dado bruto
'''


# precisaremos implementar a metodologia descrita no artigo gerando assim um primeiro dataset para treinamento
# ler o csv do dado bruto e do codebook para efetuar as transformações
raw_data_brfss = os.path.normpath(f"{notebook_dir_project_predict}{os.sep}data{os.sep}intermediate{os.sep}2023{os.sep}intermediate_{experiment_name_path}{os.sep}brfss_2023.csv")
codebook_file = os.path.normpath(f"{notebook_dir_project_predict}{os.sep}data{os.sep}intermediate{os.sep}2023{os.sep}intermediate_{experiment_name_path}{os.sep}brfss_2023_variaveis_expandidas_translated_categor_continu.csv")

df_brfss_2023 , df_codebook_2023 = read_data_and_codebook(raw_data_brfss, codebook_file)


# %% [markdown]
# Remover colunas não documentadas no codebook
# 
# Remover colunas identificadas como irrelevantes para o estudo
# 

# %%
df_brfss_2023_to_drop = df_brfss_2023.copy()

# detecção das colunas que não estão no codebook
colunas_nao_mapeadas = [col for col in df_brfss_2023_to_drop.columns if col not in df_codebook_2023['SAS Variable Name'].values]

# as colunas a seguir possuem dados de identificação de amostra 
# e não são relevantes para o modelo
irrelevant_columns = [
   'FMONTH', 
   'IDATE', 
   'IMONTH', 
   'IDAY', 
   'IYEAR', 
   'DISPCODE', 
   'SEQNO', 
   '_PSU', 
   'CTELENM1', 
   'CELPHON1',
   'CTELNUM1',
   'CELLFON5',
   'QSTVER',
   ]

df_2023_removed_columns = df_brfss_2023_to_drop.drop(columns=(colunas_nao_mapeadas + irrelevant_columns), errors='ignore')
# df_2023_removed_columns


# %% [markdown]
# Alterar para zero valores 88 e 888 no Dado 

# %%
'''import pandas as pd
from IPython.display import display

# Pré-requisito: O DataFrame 'codebook_2023' já deve estar carregado no seu ambiente.

# A linha de código principal que realiza a consulta:
# 1. Converte a coluna 'Value' para string para garantir a busca de texto.
# 2. Usa .str.contains('88') para encontrar a substring '88'.
# 3. 'na=False' garante que valores nulos (NaN) não causem erro.
linhas_com_88 = df_codebook_2023[df_codebook_2023['Value'].astype(str).str.contains('88', na=False)]

# Exibe o DataFrame resultante de forma bem formatada
print(f"Foram encontradas {len(linhas_com_88)} linhas onde 'Value' contém '88':")
display(linhas_com_88)
'''

# %%

convert_to_zero = {
'PHYSHLTH': 88,
'MENTHLTH': 88,
'POORHLTH': 88,
'STRENGTH': 888,
'CHILDREN': 88,
'FALL12MN': 88,
'FALLINJ5': 88,
'ALCDAY4': 888,
'AVEDRNK3': 88,
'DRNK3GE5': 88,
'MAXDRNKS': 88,
'DRNKDRI2': 88,
'CHKHEMO3': 88,
'LCSFIRST': 888,
'MARIJAN1': 88,
}


'''    
'JOINPAI2': numero maior que 10 deixar vazio

# Substituindo valores 88/888 por zero nas colunas especificadas

'''

# %%


convert_to_zero = {
'PHYSHLTH': 88,
'MENTHLTH': 88,
'POORHLTH': 88,
'STRENGTH': 888,
'CHILDREN': 88,
'FALL12MN': 88,
'FALLINJ5': 88,
'ALCDAY4': 888,
'AVEDRNK3': 88,
'DRNK3GE5': 88,
'MAXDRNKS': 88,
'DRNKDRI2': 88,
'CHKHEMO3': 88,
'LCSFIRST': 888,
'MARIJAN1': 88,
}

# --- Script para substituir os valores ---

# Cria uma cópia para não modificar o DataFrame original
df_number_to_zero = df_2023_removed_columns.copy()

# Itera sobre o dicionário e substitui os valores
for coluna, valor_para_substituir in convert_to_zero.items():
    # Verifica se a coluna do dicionário existe no DataFrame
    if coluna in df_number_to_zero.columns:
        print(f"Substituindo o valor {valor_para_substituir} por 0 na coluna '{coluna}'...")
        # .replace() é eficiente para esta operação
        df_number_to_zero[coluna] = df_number_to_zero[coluna].replace(valor_para_substituir, 0.0)

print("\nDataFrame Processado:")
df_number_to_zero

# %% [markdown]
# Valores no dado sem referência no codebook

# %%
'''  
Coluna: NUMADULT
Seráa deletada por ter 80% de missing data

'''

# precisaremos remover alguns valores cuidadosamente 
# que não estão devidamente docuentados no codebook 
# e estão no dado bruto



values_to_remove_not_in_codebook = {

 'LANDSEX2': [3],
 'CELLSEX2': [3],
 'CCLGHOUS': [2],

}

df_2023_to_remove_values  = df_number_to_zero.copy()

# ler a coluna na chave do dicionário e remover os valores no dado
for column, values in values_to_remove_not_in_codebook.items():
    if column in df_2023_to_remove_values.columns:
        df_2023_to_remove_values = df_2023_to_remove_values[~df_2023_to_remove_values[column].isin(values)]
    else:
        print(f"Coluna {column} não encontrada no DataFrame.")


df_2023_to_remove_values

# %% [markdown]
# Remover do dado valores que se referem a strings inválidas

# %%
# remover do dado os valores que se referem a strings inválidas no codebook 

# Quando o valor no codebook for algum desses da lista , mudaremos para vazio
df_remove_string_index = df_2023_to_remove_values.copy()

string_list_to_null = [
 "Refused",
 "Don’t know",
 "Not sure",
 "Don’t know/Not sure",
 "Refused to answer",
 "Blank",
 "Missing",
 "Not asked or Missing",

 ]

# df_withhout_invalid = limpar_valores_indesejados_codebook(df_remove_string_index, codebook_2023, string_list_to_null)

####### alteramos o argumento para substituir por vazio
df_withhout_invalid = substituir_valores_por_zero_baseado_no_codebook(df_remove_string_index, df_codebook_2023, string_list_to_null, sufixo='')
df_withhout_invalid.to_csv(f"{notebook_dir_project_predict}{os.sep}data{os.sep}intermediate{os.sep}2023{os.sep}intermediate_{experiment_name_path}{os.sep}brfss_2023_string_null.csv", index=False)
df_withhout_invalid


# %%
# Outras Normalizações aqui

# %%
'''
Recaptulando
Variáveis de interesse 
CVDINFR4 = Ever Diagnosed with Heart Attack 
CVDCRHD4 = Ever Diagnosed with Angina or Coronary Heart Disease 
_MICHD = Ever Diagnosed with Heart Disease 

'''

# assim vamos retirar os registros (linhas) onde é vazio para a variável MICHD
# removar o missing data por linha , todos que foem nulos para as variável MICHD
'''
df_remove_null_lines = df_withhout_invalid.copy()
df_remove_null_lines_result = remover_linhas_com_alvo_nulo(df_remove_null_lines, '_MICHD_processed')
df_remove_null_lines_result.to_csv(f"{notebook_dir_project_predict}{os.sep}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_michd_not_null.csv", index=False)
df_remove_null_lines_result'''

# %% [markdown]
# Remover colunas com mais de 30% de missing data

# %%
# Aplicar seleção de variáveis com menos de 30 % de missing ### PAPER

# df_brfss_2023_filtered_columns_csv = pd.read_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023.csv")
df_brfss_2023_filtered_columns_csv = df_withhout_invalid.copy()

# Calcular o percentual de valores nulos por coluna
percentual_na = df_brfss_2023_filtered_columns_csv.isnull().mean() * 100

# Selecionar colunas com menos de 30% de valores nulos - usar < 30 # conforme o paper
colunas_boas = percentual_na[percentual_na  < 30].index # Usar a proporção de missing data da variavel alvo para filtrar as colunas

# Criar novo DataFrame apenas com essas colunas
df_brfss_2023_csv_filtered = df_brfss_2023_filtered_columns_csv[colunas_boas]

# Exibir informações do DataFrame filtrado
df_brfss_2023_csv_filtered.info()

# Exibir o DataFrame
df_brfss_2023_csv_filtered

# %% [markdown]
# Removendo os registros que possuem algum missing data

# %%
'''  
Outros tratamentos de strings e dados de data
Tratamento de missing data (30% conforme o paper)
Tratamento de variáveis contínuas

'''
# a amostra onde a linha tiver mais de 30% de missing data


def remover_linhas_com_muitos_nulos(df, limiar_percentual_nulos_linha=0):
    """
    Remove linhas de um DataFrame que excedem um determinado limiar
    percentual de valores ausentes (NaN).

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        limiar_percentual_nulos_linha (float): Limiar percentual (0-100).
                                                Linhas com mais % de NaNs do que
                                                este valor serão removidas.
                                                Default é 30.0 (30%).

    Retorno:
        pd.DataFrame: DataFrame com as linhas problemáticas removidas.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("A entrada 'df' deve ser um DataFrame Pandas.")
    if not (0 <= limiar_percentual_nulos_linha <= 100):
        raise ValueError("O 'limiar_percentual_nulos_linha' deve estar entre 0 e 100.")

    print(f"DataFrame original - Shape: {df.shape}")

    # Calcula o número mínimo de valores não nulos que uma linha deve ter
    # Se uma linha tiver menos que isso, ela tem mais do que o limiar de nulos
    min_valores_nao_nulos_por_linha = int(df.shape[1] * (1 - (limiar_percentual_nulos_linha / 100.0)))
    
    # Garante que o mínimo não seja negativo se o limiar for 100%
    min_valores_nao_nulos_por_linha = max(0, min_valores_nao_nulos_por_linha)

    print(f"Limiar para remoção: Linhas com mais de {limiar_percentual_nulos_linha}% de valores nulos.")
    print(f"Isso significa que uma linha deve ter pelo menos {min_valores_nao_nulos_por_linha} valores não nulos (de {df.shape[1]} colunas).")

    # df.dropna() com o parâmetro 'thresh' mantém linhas com pelo menos 'thresh' valores não nulos.
    df_filtrado = df.dropna(thresh=min_valores_nao_nulos_por_linha, axis=0) # axis=0 para operar nas linhas

    print(f"DataFrame após remoção de linhas - Shape: {df_filtrado.shape}")
    print(f"Número de linhas removidas: {df.shape[0] - df_filtrado.shape[0]}")
    
    return df_filtrado



# %%
df_to_filter_samples = df_brfss_2023_csv_filtered.copy()

df_sem_linhas_com_muitos_nulos = remover_linhas_com_muitos_nulos(df_to_filter_samples, 
                                                                 limiar_percentual_nulos_linha=0)
# precisa criar a pasta do experimento
df_sem_linhas_com_muitos_nulos.to_csv(f"{notebook_dir_project_predict}{os.sep}data{os.sep}output{os.sep}2023{os.sep}output_{experiment_name_path}{os.sep}brfss_2023_cleaned_to_model.csv", index=False)


df_sem_linhas_com_muitos_nulos                            


# %% [markdown]
# Vamos  separar as colunas  que devem passar por Standard Scaler ou One Hot Encoding

# %%
# colunas do dado 
data_cleaned_to_model = pd.read_csv(f"{notebook_dir_project_predict}{os.sep}data{os.sep}output{os.sep}2023{os.sep}output_{experiment_name_path}{os.sep}brfss_2023_cleaned_to_model.csv")
list_columns_df_to_model = data_cleaned_to_model.columns
list_columns_df_to_model

# %%
# codebook 
codebook_file_final =  pd.read_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}intermediate_{experiment_name_path}{os.sep}brfss_2023_variaveis_expandidas_translated_categor_continu.csv")
codebook_file_final


# %%
# traga de codebook_file_final os registros onde a var sas name está em list_columns_df_to_model
codebook_filtered = codebook_file_final[codebook_file_final['SAS Variable Name'].isin(list_columns_df_to_model)]
# retorne a primeira ocorrencia de cada variável sas name
codebook_filtered = codebook_filtered.drop_duplicates(subset=['SAS Variable Name'], keep='first')
codebook_filtered

# %%
# podemos separar aqui as colunas que serão dummyizadas
# de codebook_filtered salve em duas listas de variavel sas name onde a coluna type of value é continuou ou categorical

list_sas_name_continuous = codebook_filtered[codebook_filtered['Type of Value'] == 'continuous']['SAS Variable Name'].tolist()
list_sas_name_categorical = codebook_filtered[codebook_filtered['Type of Value'] == 'categorical']['SAS Variable Name'].tolist()

# variáveis contínuas têm " - " no value 

# print(list_sas_name_continuous)
print(len(list_sas_name_continuous))
# print(list_sas_name_categorical)
print(len(list_sas_name_categorical))




# %%
# vamos varificar a quantidade de  entradas das variaveis categoricas 
# no codebook_file_final conte a quantidade de ocorrencias de cada item de list_sas_name_categorical em sas variavek name

for var in list_sas_name_categorical:
    count = codebook_file_final[codebook_file_final['SAS Variable Name'] == var].shape[0]
    print(f"Variável: {var}, Ocorrências no codebook: {count}")


# %%
# aplicar o Standard Scale  em variáveis contínuas e binarização em variáveis categóricas

# salvar na pasta do experimento os datasets X, Y , X_train , X_test , y_train, y_test



# %% [markdown]
# Continua em Data Models 

# %% [markdown]
# # Explore Scripts

# %%

# Normalizar variáveis contínuas
# xistem variávis onde por eemplo a prática de atividades físicas ora aparece como vezes por semana  ou mes


'''     
Coluna: HHADULT OK

----------------------------------------
Coluna: PHYSHLTH OK

----------------------------------------
Coluna: MENTHLTH OK

----------------------------------------
'''

# verificar os valor que estão como dias , vezs por semana, meses, anos e etc
# Norrmalzar certos dados 

# multicolinearidade
# scikit learn RFECV


# %%


# %% [markdown]
# Abaixo função para mapear códigos no dado sem referência no codebook

# %%


# identificar se nas colunas restantes ignorando o valores vazios tem algum valor não representado no codebook e listar essas colunas e o valor presente no dado que não está no codebook
def encontrar_e_salvar_valores_nao_mapeados(df_brfss, codebook_df, caminho_arquivo_saida):
    """
    Identifica, para cada coluna do DataFrame, os valores que não possuem
    um 'Value Label' correspondente no codebook. Valores nulos (NaN)
    no DataFrame de dados são ignorados. Os resultados são salvos em um arquivo de texto.

    Parâmetros:
        df_brfss (pd.DataFrame): DataFrame original com os dados.
        codebook_df (pd.DataFrame): DataFrame do codebook com 'SAS Variable Name',
                                    'Value', e 'Value Label'.
        caminho_arquivo_saida (str): Caminho completo para o arquivo .txt onde os
                                     resultados serão salvos.

    Retorno:
        dict: Um dicionário onde as chaves são nomes de colunas e os valores
              são listas de valores únicos daquela coluna que não foram
              encontrados nos códigos ('Value') do codebook para aquela variável.
              Retorna apenas colunas que tiveram valores não mapeados.
              Retorna um dicionário vazio se nenhum valor não mapeado for encontrado.
    """

    def limpar_valor_para_comparacao(v):
        """Limpa e converte valor para string para comparação com os códigos do codebook."""
        if pd.isna(v): # Se o valor já for NaN, não há como limpar para string de forma útil aqui.
            return None # Será filtrado depois pelo dropna() nos valores únicos da coluna.
        try:
            return str(int(float(v)))
        except ValueError:
            return str(v).strip()
        except Exception:
            return str(v).strip()

    valores_nao_encontrados_geral = {}

    with tqdm(total=len(df_brfss.columns), desc="Verificando colunas") as pbar:
        for coluna in df_brfss.columns:
            pbar.update(1)
            
            codebook_var_atual = codebook_df[codebook_df['SAS Variable Name'] == coluna]
            
            if codebook_var_atual.empty:
                continue

            codigos_no_codebook_para_coluna = set(
                codebook_var_atual['Value'].dropna().apply(limpar_valor_para_comparacao)
            )

            if not codigos_no_codebook_para_coluna:
                continue
                
            # Pega os valores únicos da coluna no DataFrame de dados, IGNORANDO NaNs
            valores_unicos_na_coluna_dados = df_brfss[coluna].dropna().unique()
            
            valores_nao_encontrados_nesta_coluna = set()

            for valor_dados in valores_unicos_na_coluna_dados:
                valor_dados_limpo = limpar_valor_para_comparacao(valor_dados)
                
                # Se valor_dados_limpo for None (era NaN originalmente), não deve ser comparado
                if valor_dados_limpo is None:
                    continue

                if valor_dados_limpo not in codigos_no_codebook_para_coluna:
                    valores_nao_encontrados_nesta_coluna.add(valor_dados) 
            
            if valores_nao_encontrados_nesta_coluna:
                # Converte para lista e ordena para consistência na saída
                # Garante que os valores sejam strings para evitar problemas de tipo misto no sort
                valores_nao_encontrados_geral[coluna] = sorted(list(map(str, valores_nao_encontrados_nesta_coluna)))
    
    # Salvar os resultados no arquivo de texto
    try:
        with open(caminho_arquivo_saida, 'w', encoding='utf-8') as f:
            if valores_nao_encontrados_geral:
                f.write("Valores não encontrados no codebook (ignorando NaNs nos dados):\n")
                f.write("="*60 + "\n")
                for coluna, valores in valores_nao_encontrados_geral.items():
                    f.write(f"Coluna: {coluna}\n")
                    f.write(f"Valores não mapeados: {valores}\n")
                    f.write("-" * 40 + "\n")
                print(f"\nResultados salvos em: {caminho_arquivo_saida}")
            else:
                f.write("Nenhum valor não mapeado encontrado (ignorando NaNs nos dados).\n")
                print(f"\nNenhum valor não mapeado encontrado. Arquivo salvo em: {caminho_arquivo_saida}")
    except IOError as e:
        print(f"Erro ao salvar o arquivo em '{caminho_arquivo_saida}': {e}")
        # Retorna o dicionário mesmo se o salvamento falhar, para não perder os dados
        return valores_nao_encontrados_geral
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao tentar salvar o arquivo: {e}")
        return valores_nao_encontrados_geral
                
    return valores_nao_encontrados_geral

caminho_arquivo_saida = os.path.normpath(f"{notebook_dir_project_predict}{os.sep}data{os.sep}intermediate{os.sep}2023{os.sep}{experiment_name_path}{os.sep}valores_nao_mapeados_arquivo_tratado.txt")
valores_nao_mapeados = encontrar_e_salvar_valores_nao_mapeados(df_sem_linhas_com_muitos_nulos, df_codebook_2023, caminho_arquivo_saida)





# %% [markdown]
# Mapear codebook e gerar um novo dataset

# %%
# para gerar um dado mapeado , precisaremos de mais cautela
# pois alguns valores não coincidem com o codebook ou coincidem de forma não padronizada

'''df_2023_to_treat = df_brfss_2023.copy() # há possívelmente um erro em mapear simplesmente pois alguns valores podem ter mais de uma representação
# df_cleaned_date = limpar_colunas_data(df_2023_to_treat)

df_map = mapear_colunas_para_rotulo(df_2023_to_treat, df_codebook_2023)
df_map.to_csv(f"{notebook_dir_project_predict}{os.sep}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_maped.csv")
df_map'''

# %% [markdown]
# Dataset final tratado para uso nos modelos

# %%
# balanceamento dos dados
from sklearn.utils import resample
def balancear_dados_por_coluna(df, coluna_alvo, metodo='oversample', proporcao=1.0):
    """
    Balanceia o DataFrame com base na coluna alvo usando oversampling ou undersampling.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser balanceado.
        coluna_alvo (str): Nome da coluna alvo para balanceamento.
        metodo (str): Método de balanceamento ('oversample' ou 'undersample').
        proporcao (float): Proporção desejada entre as classes após o balanceamento.

    Retorno:
        pd.DataFrame: DataFrame balanceado.
    """
    if metodo not in ['oversample', 'undersample']:
        raise ValueError("Método deve ser 'oversample' ou 'undersample'.")

    # Conta as ocorrências de cada classe na coluna alvo
    contagem_classes = df[coluna_alvo].value_counts()
    classe_mais_frequente = contagem_classes.idxmax()
    tamanho_classe_mais_frequente = contagem_classes.max()

    if metodo == 'oversample':
        # Oversampling da classe minoritária
        df_minority = df[df[coluna_alvo] != classe_mais_frequente]
        df_majority = df[df[coluna_alvo] == classe_mais_frequente]

        # Calcula quantas vezes precisamos replicar a classe minoritária
        n_replicas = int(tamanho_classe_mais_frequente * proporcao / len(df_minority))

        # Realiza o oversampling
        df_minority_oversampled = resample(df_minority, 
                                           replace=True, 
                                           n_samples=len(df_minority) * n_replicas, 
                                           random_state=42)

        # Combina as classes
        df_balanceado = pd.concat([df_majority, df_minority_oversampled])

    elif metodo == 'undersample':
        # Undersampling da classe majoritária
        df_majority = df[df[coluna_alvo] == classe_mais_frequente]
        df_minority = df[df[coluna_alvo] != classe_mais_frequente]

        # Calcula quantas amostras manteremos da classe majoritária
        n_samples_majority = int(len(df_minority) * proporcao)

        # Realiza o undersampling
        df_majority_undersampled = resample(df_majority, 
                                               replace=False,
                                                  n_samples=n_samples_majority, 
                                                  random_state=42)

# %%
# Dummie nos dados categóricos


