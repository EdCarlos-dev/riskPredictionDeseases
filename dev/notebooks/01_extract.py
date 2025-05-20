# %% [markdown]
# Risco de Doenças Cardíacas baseado em variáveis ambientais e sociodemográficas de dados públicos norte americanos

# %% [markdown]
# https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data
# 
# 

# %% [markdown]
# https://github.com/kamilpytlak/data-science-projects/blob/main/heart-disease-prediction/2022/notebooks/data_processing.ipynb

# %% [markdown]
# https://www.kaggle.com/code/alphiree/cvds-risk-prediction-notebook-full

# %% [markdown]
# https://www.kaggle.com/code/aemreusta/heart-attack-prediction

# %% [markdown]
# https://www.kaggle.com/code/mhmedgaber/heart-diesese-detection-4-models-95/notebook

# %% [markdown]
# https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/code

# %% [markdown]
# Prediction 
# APP and Article created referrences 
# 
# https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset

# %% [markdown]
# # LIBS
# 

# %%
# libs
import os
import json
import zipfile
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from io import BytesIO
import plotly.express as px
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator

# docling
# Mage AI

# %% [markdown]


# Função para buscar `var_info` de uma variável específica
def consultar_var_info(json_file_path, sas_variable_name):

    # Carregar o JSON
    with open(json_file_path, "r", encoding="utf-8") as f:
        variables_data = json.load(f)

    for variable in variables_data:
        if variable["SAS Variable Name"] == sas_variable_name:
            return variable["var_info"]
        
    return None  # Retorna None se a variável não for encontrada

# Função para buscar `var_info` de uma variável específica
def consultar_var_data_features(json_file_path, sas_variable_name, sas_variable_feature):

    # Carregar o JSON
    with open(json_file_path, "r", encoding="utf-8") as f:
        variables_data = json.load(f)

    for variable in variables_data:
        if variable["SAS Variable Name"] == sas_variable_name:
            return variable[sas_variable_feature]
        
    return None  # Retorna None se a variável não for encontrada


def contar_com_rotulo(
    df_brfss: pd.DataFrame,
    codebook_df: pd.DataFrame,
    coluna: str
) -> pd.Series:
    
    ''' 
    Substitui valores e nomes de variável com base no codebook e retorna o value_counts com rótulos legíveis.
    
    Parâmetros:
    - df_brfss: DataFrame com os dados brutos (valores numéricos)
    - df_codebook: DataFrame extraído do codebook
    - var_code: Nome da variável SAS no dataframe original (ex: 'CVDINFR4')
    
    Retorna:
    - Series com value_counts() indexado pelo valor legível
    
    '''
    # Verifica se a coluna existe no DataFrame
    if coluna not in df_brfss.columns:
        print(f"Coluna '{coluna}' não encontrada no DataFrame BRFSS.")
        return pd.Series(dtype=int)

    # Filtra o codebook para a variável
    codebook_var = codebook_df[codebook_df['SAS Variable Name'] == coluna]
    if codebook_var.empty:
        print(f"Variável '{coluna}' não encontrada no codebook.")
        return df_brfss[coluna].value_counts(dropna=False)

    # Remove valores nulos da coluna 'Value' do codebook
    codebook_var = codebook_var.dropna(subset=['Value'])

    # Cria dicionário de mapeamento: valor (como string) -> legenda
    mapa_valores = dict(zip(
        codebook_var['Value'].astype(str).str.strip(),
        codebook_var['Value Label']
    ))

    # Converte valores do BRFSS para strings compatíveis com o dicionário
    def limpar_valor(v):
        if pd.isna(v):
            return "BLANK"  # ou "Desconhecido"
        try:
            return str(int(v))  # ex: 1.0 -> '1'
        except:
            return str(v).strip()

    serie_convertida = df_brfss[coluna].apply(limpar_valor)
    serie_traduzida = serie_convertida.map(mapa_valores).fillna("Desconhecido")

    # Nome descritivo da variável
    nome_variavel = codebook_var['Label'].dropna().iloc[0] if not codebook_var['Label'].dropna().empty else coluna
    serie_traduzida.name = nome_variavel

    # Retorna a contagem com rótulos legíveis
    return serie_traduzida.value_counts(dropna=False)



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
        codebook_var['Value Label']
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


# %%
# mapeando o diretório do projeto e do arquivo notebook 
diretorio_atual_projeto = os.getcwd() # Diretório atual do arquivo
notebook_dir_project_predict = os.path.normpath(f"{diretorio_atual_projeto}{os.sep}..{os.sep}..{os.sep}") + os.sep # Diretório do projeto
print(diretorio_atual_projeto)
print(notebook_dir_project_predict)

# %% [markdown]
# Behavioral Risk Factor Surveillance System

# %% [markdown]
# Links Importantes
# 
# Behavioral Risk Factor Surveillance System
# 
# Public health surveys of 400k people from 2011-2015
# 
# https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system
# 
# https://www.kaggle.com/code/shubhamshukla11/behavioral-risk-factor-surveillance-system/notebook
# 

# %%
'''
def resumir_variavel_com_codebook(df_brfss, df_codebook, var_code):
    """
    Substitui valores e nomes de variável com base no codebook e retorna o value_counts com rótulos legíveis.
    
    Parâmetros:
    - df_brfss: DataFrame com os dados brutos (valores numéricos)
    - df_codebook: DataFrame extraído do codebook
    - var_code: Nome da variável SAS no dataframe original (ex: 'CVDINFR4')
    
    Retorna:
    - Series com value_counts() indexado pelo valor legível
    """
    # Verifica se a variável existe no DataFrame
    if var_code not in df_brfss.columns:
        print(f"Variável '{var_code}' não encontrada no DataFrame BRFSS.")
        return pd.Series(dtype=int)

    # Filtra o codebook para a variável
    subset = df_codebook[df_codebook['SAS Variable Name'] == var_code]
    if subset.empty:
        print(f"Variável '{var_code}' não encontrada no codebook.")
        return df_brfss[var_code].value_counts(dropna=False)

    # Remove valores nulos da coluna 'Value' do codebook
    subset = subset.dropna(subset=['Value'])

    # Cria dicionário de mapeamento: valor (como string) -> legenda
    mapa_valores = dict(zip(
        subset['Value'].astype(str).str.strip(),
        subset['Value Label']
    ))

    # Converte os valores da coluna no BRFSS em string no formato compatível
    def padronizar_valor(v):
        if pd.isna(v):
            return "BLANK"
        try:
            return str(int(v))  # ex: 1.0 → '1'
        except:
            return str(v).strip()

    serie_convertida = df_brfss[var_code].apply(padronizar_valor)
    serie_legenda = serie_convertida.map(mapa_valores).fillna("Desconhecido")

    # Nome descritivo da variável
    nome_legivel = subset['Label'].dropna().iloc[0] if not subset['Label'].dropna().empty else var_code
    serie_legenda.name = nome_legivel

    # Retorna a contagem com rótulos legíveis
    return serie_legenda.value_counts(dropna=False)


resumo = resumir_variavel_com_codebook(
    df_brfss_2023_csv,
    df_csv_path_expand_vars_translates,
    'CVDINFR4'
)
print(resumo)'''

# %%
# carregar os anos dosponíveis 


# %% [markdown]
# # BRFSS 2023

# %% [markdown]
# https://www.cdc.gov/brfss/annual_data/annual_2023.html

# %% [markdown]
# The 2023 BRFSS data continue to reflect the changes initially made in 2011 for weighting methodology (raking) and adding cell-phone-only respondents. The aggregate BRFSS combined landline and cell phone data set is built from the landline and cell phone data submitted for 2023 and includes data from 48 states, the District of Columbia, Guam, Puerto Rico, and the US Virgin Islands. During 2023, Kentucky and Pennsylvania were unable to collect enough data to meet the minimum requirements to be included in this public data set. The data set has been modified to comply with President Trump’s executive orders. There are some missing values which may appear to be inconsistencies in the data based on a respondents’ answers to questions that were removed.

# %% [markdown]
# Data

# %% [markdown]
# Ler dado bruto e converter para csv

# %% [markdown]
# ## 1. Obtenção do dado

# %% [markdown]
# ### 1.1 Código para obter o BRFSS 2023 do site do CDC

# %%
'''
# URL do arquivo ZIP que contém o .XPT
url = 'https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip'


# Baixar e extrair o arquivo .XPT do ZIP
response = requests.get(url)
with zipfile.ZipFile(BytesIO(response.content)) as z:
    xpt_filename = [f for f in z.namelist() if f.endswith('.XPT ')][0]
    with z.open(xpt_filename) as xpt_file:
        df_2023_from_url = pd.read_sas(xpt_file, format='xport')

############ Salvar como CSV descomentar se necessário
input_dir = f'{notebook_dir_project_predict}data{os.sep}input{os.sep}2023'

# criar  odiretorio caso não exista
if not os.path.exists(input_dir):
    print(f"Diretório {input_dir} não existe. Criando...")


os.makedirs(input_dir, exist_ok=True)
csv_path = os.path.join(input_dir, 'from_url_brfss_2023.csv')
df_2023_from_url.to_csv(csv_path, index=False)
print(f"Arquivo CSV salvo em: {csv_path}")

'''


# Defina o caminho base do projeto, se ainda não estiver definido
# notebook_dir_project_predict = '/caminho/para/seu/projeto/'

# URL do arquivo ZIP que contém o .XPT
url = 'https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip'

# Diretórios de destino relativos ao diretório base do projeto
xpt_output_dir = f'{notebook_dir_project_predict}data{os.sep}input{os.sep}2023'
csv_output_dir = f'{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023'

# Criar diretórios, se não existirem
os.makedirs(xpt_output_dir, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

# Baixar e extrair o arquivo .XPT do ZIP
response = requests.get(url)
with zipfile.ZipFile(BytesIO(response.content)) as z:
    xpt_filename = [f for f in z.namelist() if f.strip().lower().endswith('.xpt')][0]
    
    # Salvar o arquivo .XPT em data/input/2023/
    xpt_path = os.path.join(xpt_output_dir, os.path.basename(xpt_filename.strip()))
    with open(xpt_path, 'wb') as f_out:
        f_out.write(z.read(xpt_filename))
    print(f"Arquivo XPT salvo em: {xpt_path}")
    
    # Ler o conteúdo do .XPT para dataframe
    with z.open(xpt_filename) as xpt_file:
        df_2023_from_url = pd.read_sas(xpt_file, format='xport')

# Salvar como CSV em data/intermediate/2023/
csv_path = os.path.join(csv_output_dir, 'brfss_2023.csv')
df_2023_from_url.to_csv(csv_path, index=False)
print(f"Arquivo CSV salvo em: {csv_path}")


# %% [markdown]
# ### 1.2 Código para consumir o dado BRFSS 2023 baixado localmente

# %%
# Consumindo Localmente o arquivo XPT
# Caminho para o arquivo XPT (extraído do ZIP)
arquivo_xpt = f"{notebook_dir_project_predict}data{os.sep}input{os.sep}2023{os.sep}LLCP2023.XPT"

# Carregar o arquivo no pandas
df_2023 = pd.read_sas(arquivo_xpt, format='xport')

# criar arquivo csv
# df_2023.to_csv(f'{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023.csv', index=False)

df_2023

# %% [markdown]
# ## 2. Exibir informações gerais do dado

# %%
df_2023_csv = pd.read_csv(f'{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023.csv')

# salve num txt a saida do .info()
with open(f'{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_info.txt', 'w') as f:
    df_2023_csv.info(buf=f)


# %% [markdown]
# ## 3, Arquivo CODEBOOK das Variáveis

# %% [markdown]
# ### 3.1 Consumindo da WEB para CSV

# %%
# URL do arquivo ZIP que contém o .XPT
url_codebook = 'https://www.cdc.gov/brfss/annual_data/2023/zip/codebook23_llcp-v2-508.zip'


# Baixar e extrair o arquivo .XPT do ZIP
response_url_codebook = requests.get(url_codebook)
with zipfile.ZipFile(BytesIO(response_url_codebook.content)) as z:
    html_filename = [f for f in z.namelist() if f.endswith('.HTML')][0]
    with z.open(html_filename, "r") as html_file:
        html_content = html_file.read()

soup = BeautifulSoup(html_content, 'html.parser')

# Lista para armazenar os dados extraídos
dados = []

# Encontrar todos os <td> que contêm as informações (ajuste o seletor se necessário)
tds = soup.find_all('td', class_='l m linecontent')

for td in tds:
    texto_bruto = td.get_text(separator="\n").strip()
    linhas = texto_bruto.split("\n")
    dado = {}

    for linha in linhas:
        if ":" in linha:
            chave, valor = linha.split(":", 1)
            chave = chave.strip().replace("\xa0", " ")  # Remove &nbsp;
            valor = valor.strip().replace("\xa0", " ")
            dado[chave] = valor

    # Adiciona o dicionário de cada variável na lista
    dados.append(dado)

# Transforma a lista de dicionários em DataFrame
df_vars_2023 = pd.DataFrame(dados)

# Se quiser salvar para Excel/CSV
# df_vars_2023.to_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis.csv", index=False)


# %% [markdown]
# ### 3.2 Consumindo Localmente para CSV

# %%

# Suponha que você tenha carregado o HTML completo em uma variável chamada html_content
# Exemplo de leitura de um arquivo local
with open(f"{notebook_dir_project_predict}data{os.sep}input{os.sep}2023{os.sep}codebook23_llcp-v2-508{os.sep}USCODE23_LLCP_021924.HTML", "r", encoding="utf-8") as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')

# Lista para armazenar os dados extraídos
dados = []

# Encontrar todos os <td> que contêm as informações (ajuste o seletor se necessário)
tds = soup.find_all('td', class_='l m linecontent')

for td in tds:
    texto_bruto = td.get_text(separator="\n").strip()
    linhas = texto_bruto.split("\n")
    dado = {}

    for linha in linhas:
        if ":" in linha:
            chave, valor = linha.split(":", 1)
            chave = chave.strip().replace("\xa0", " ")  # Remove &nbsp;
            valor = valor.strip().replace("\xa0", " ")
            dado[chave] = valor

    # Adiciona o dicionário de cada variável na lista
    dados.append(dado)

# Transforma a lista de dicionários em DataFrame
df_vars_2023 = pd.DataFrame(dados)


# Se quiser salvar para Excel/CSV
# df_vars_2023.to_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis.csv", index=False)



# %%
''' 
O Codebook tem apenas 344 variaveis, enquanto o dataframe tem 350 colunas.
Isso pode ocorrer por algumas razões:

1. **Variáveis não documentadas**: Algumas variáveis podem não estar documentadas no codebook, mas ainda existem no conjunto de dados.
2. **Variáveis temporárias ou de controle**: Algumas variáveis podem ser criadas para controle interno ou para processamento, mas não são relevantes para a análise final.
3. **Mudanças entre versões**: Se o conjunto de dados foi atualizado ou modificado, pode haver novas variáveis que não estão refletidas no codebook.
4. **Erros de documentação**: Pode haver erros ou omissões no codebook que não refletem com precisão todas as variáveis presentes no conjunto de dados.
5. **Variáveis de metadados**: Algumas colunas podem conter informações adicionais ou metadados que não são considerados variáveis principais.
6. **Variáveis de controle de qualidade**: Algumas variáveis podem ser usadas para verificar a qualidade dos dados ou para validação, mas não são relevantes para a análise principal.
7. **Variáveis de identificação**: Algumas colunas podem ser usadas para identificar registros ou participantes, mas não são consideradas variáveis de interesse para análise.
# 8. **Variáveis de agrupamento**: Algumas variáveis podem ser usadas para agrupar ou categorizar dados, mas não são relevantes para a análise principal.
# 9. **Variáveis de data/hora**: Algumas colunas podem conter informações de data ou hora que não são consideradas variáveis principais.
10. **Variáveis de resposta**: Algumas colunas podem conter respostas a perguntas específicas, mas não são consideradas variáveis principais para análise.
# 11. **Variáveis de controle de amostra**: Algumas colunas podem ser usadas para controlar a amostra ou o tamanho da amostra, mas não são relevantes para a análise principal.
# 12. **Variáveis de codificação**: Algumas colunas podem conter códigos ou identificadores que não são considerados variáveis principais para análise.
# 13. **Variáveis de resposta múltipla**: Algumas colunas podem conter respostas a perguntas de múltipla escolha, mas não são consideradas variáveis principais para análise.
# 14. **Variáveis de resposta aberta**: Algumas colunas podem conter respostas abertas que não são consideradas variáveis principais para análise.
# 15. **Variáveis de controle de qualidade**: Algumas colunas podem ser usadas para verificar a qualidade dos dados ou para validação, mas não são relevantes para a análise principal.
# 16. **Variáveis de controle de amostra**: Algumas colunas podem ser usadas para controlar a amostra ou o tamanho da amostra, mas não são relevantes para a análise principal.


'''

# %% [markdown]
# ### 3.3 Adicionar coluna com a tradução

# %%

# Copiar o dataframe df_vars_2023
df_vars_2023_translated = pd.read_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis.csv")

# Função para traduzir texto
def traduzir_texto(texto, src_lang='en', target_lang='pt'):
    try:
        return GoogleTranslator(source=src_lang, target=target_lang).translate(texto)
    except Exception as e:
        print(f"Erro ao traduzir: {e}")
        return texto

# Adicionar coluna 'question_translate' com a tradução da coluna 'Question'
tqdm.pandas()  # Inicializa o tqdm para pandas
df_vars_2023_translated['question_translate'] = df_vars_2023_translated['Question'].progress_apply(lambda x: traduzir_texto(x))
df_vars_2023_translated['label_translate'] = df_vars_2023_translated['Label'].progress_apply(lambda x: traduzir_texto(x))
df_vars_2023_translated['section_name_translate'] = df_vars_2023_translated['Section Name'].progress_apply(lambda x: traduzir_texto(x))

# copiar o dataframe df_vars_2023 e adicionar question_translate com a tradução da string da coluna question
df_vars_2023_translated.to_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis_translated.csv", index=False)


# %% [markdown]
# ### 3,4 CODEBOOK Completo em JSON

# %%
# Extração de todos dos dados do codebook html para um formato de dados json

from bs4 import BeautifulSoup
import pandas as pd
import json


# dado acima precisa ser complmentado com o resto dos detalhes das variáveis das colunas do dado de 2023
# mapeando as informaçoes do codebook se obtêm o seguinte estrutura de dados

# Ao mapear o arquivo do CODEBOOK temos as seguintes informações da variável
'''variable_data = {
    'Label': '',
    'Section Name': '',
    'Section Number': '',
    'Question Number': '',
    'Column': '',
    'Type of Variable': '',
    'SAS Variable Name': '',
    'Question Prologue': '',
    'Question': '',
    'var_info': {
            'Value': [],
            'Value Label': [],
            'Frequency': [],
            'Percentage': [],
            'Weighted Percentage': [],

    }
}
'''


# Caminho do arquivo HTML
html_path = f"{notebook_dir_project_predict}data{os.sep}input{os.sep}2023{os.sep}codebook23_llcp-v2-508{os.sep}USCODE23_LLCP_021924.HTML"

# Ler o HTML
with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')

# Lista para armazenar os dados extraídos
variables_list = []

# Encontrar todas as tabelas no HTML
tables = soup.find_all("table")

for table in tables:
    # Captura a célula com os metadados (ajuste a classe se necessário)
    td = table.find('td', class_='l m linecontent')
    
    if td:
        # Captura os metadados
        texto_bruto = td.get_text(separator="\n").strip()
        linhas = texto_bruto.split("\n")
        
        variable_data = {
            'Label': '',
            'Section Name': '',
            'Section Number': '',
            'Question Number': '',
            'Column': '',
            'Type of Variable': '',
            'SAS Variable Name': '',
            'Question Prologue': '',
            'Question': '',
            'var_info': {
                'Value': [],
                'Value Label': [],
                'Frequency': [],
                'Percentage': [],
                'Weighted Percentage': []
            }
        }

        # Preenchendo os metadados
        for linha in linhas:
            if ":" in linha:
                chave, valor = linha.split(":", 1)
                chave = chave.strip().replace("\xa0", " ")  # Remove &nbsp;
                valor = valor.strip().replace("\xa0", " ")
                if chave in variable_data:
                    variable_data[chave] = valor

        # Nome da variável SAS
        var_name = variable_data['SAS Variable Name']
        if not var_name:
            continue  # Ignorar se não tem nome de variável SAS

        # Captura os cabeçalhos da tabela de valores
        headers = [th.text.strip() for th in table.find_all("th")]
        
        # Captura os valores da tabela
        for row in table.find_all("tr")[1:]:  # Pula cabeçalhos
            cols = [td.text.strip() for td in row.find_all("td")]
            if cols and len(cols) == len(headers):  # Evita erro de indexação
                for i, col in enumerate(cols):
                    header = headers[i]
                    if header in variable_data['var_info']:
                        variable_data['var_info'][header].append(col)

        # Adiciona à lista
        variables_list.append(variable_data)

# Converter para DataFrame (caso queira trabalhar com tabelas)
df_vars_2023 = pd.DataFrame(variables_list)

# Salvar CSV e JSON para consulta posterior
# df_vars_2023.to_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}brfss_2023_variaveis.csv", index=False)

json_vars_path_file = f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis.json"
with open(json_vars_path_file, "w", encoding="utf-8") as f:
    json.dump(variables_list, f, indent=4, ensure_ascii=False)

print("Extração concluída. Dados salvos em CSV e JSON.")





# %% [markdown]
# ### 3.5 CODEBOOK Completo em CSV

# %%
# Chaves fixas que queremos manter
CHAVES_FIXAS = [
    'Label',
    'Section Name',
    'Section Number',
    'Question Number',
    'Column',
    'Type of Variable',
    'SAS Variable Name',
    'Question Prologue',
    'Question'
]

CHAVES_VAR_INFO = [
    'Value',
    'Value Label',
    'Frequency',
    'Percentage',
    'Weighted Percentage'
]

def expandir_variavel_em_tabela_filtrada(dado_var):
    """Expande uma variável, mantendo só as chaves desejadas."""
    var_info = dado_var.get("var_info", {})
    n = len(var_info.get("Value", []))  # Número de valores a expandir

    # Filtros só com chaves que queremos
    fixos_filtrados = {k: [dado_var.get(k, '')] * n for k in CHAVES_FIXAS}
    var_info_filtrado = {k: var_info.get(k, [''] * n) for k in CHAVES_VAR_INFO}

    # Combinar tudo
    fixos_filtrados.update(var_info_filtrado)
    return pd.DataFrame(fixos_filtrados)

def carregar_json_com_filtro(json_path):
    """Lê o JSON e transforma em uma única tabela filtrada."""
    with open(json_path, "r", encoding="utf-8") as f:
        dados = json.load(f)

    tabelas = [expandir_variavel_em_tabela_filtrada(var) for var in dados]
    return pd.concat(tabelas, ignore_index=True)


json_path = f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis.json"
df_resultado = carregar_json_com_filtro(json_path)

df_resultado.to_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis_expandidas.csv", index=False)



# %% [markdown]
# ### 3.6 Adicionar colunas com a tradução

# %%

# Copiar o dataframe df_vars_2023
df_vars_2023_expand_translated = pd.read_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis_expandidas.csv")

# Função para traduzir texto
def traduzir_texto(texto, src_lang='en', target_lang='pt'):
    try:
        return GoogleTranslator(source=src_lang, target=target_lang).translate(texto)
    except Exception as e:
        print(f"Erro ao traduzir: {e}")
        return texto

# Adicionar coluna 'question_translate' com a tradução da coluna 'Question'
tqdm.pandas()  # Inicializa o tqdm para pandas
df_vars_2023_expand_translated['label_translate'] = df_vars_2023_expand_translated['Label'].progress_apply(lambda x: traduzir_texto(x))
df_vars_2023_expand_translated['section_name_translate'] = df_vars_2023_expand_translated['Section Name'].progress_apply(lambda x: traduzir_texto(x))
df_vars_2023_expand_translated['question_translate'] = df_vars_2023_expand_translated['Question'].progress_apply(lambda x: traduzir_texto(x))
df_vars_2023_expand_translated['value_label_translate'] = df_vars_2023_expand_translated['Value Label'].progress_apply(lambda x: traduzir_texto(x))

# copiar o dataframe df_vars_2023 e adicionar question_translate com a tradução da string da coluna question
df_vars_2023_expand_translated.to_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis_expandidas_translated.csv", index=False)


# %%
df_vars_2023_expand_translated_read = pd.read_csv(f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis_expandidas_translated.csv")
df_vars_2023_expand_translated_read

# %% [markdown]
# Retornar uma fração do Codebook no formato JSON ou CSV 

# %% [markdown]
# ### 3.7 fração do Codebook JSON

# %%
'''variable_data = {
    'Label': '',
    'Section Name': '',
    'Section Number': '',
    'Question Number': '',
    'Column': '',
    'Type of Variable': '',
    'SAS Variable Name': '',
    'Question Prologue': '',
    'Question': '',
    'var_info': {
            'Value': [],
            'Value Label': [],
            'Frequency': [],
            'Percentage': [],
            'Weighted Percentage': [],

    }
}
'''

age_var = 'CVDSTRK3'
feature_var = 'var_info'

# Caminho do JSON gerado
json_path = f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis.json"
 # Substitua pelo nome da variável desejada
var_info = consultar_var_data_features(json_file_path=json_path, 
                                       sas_variable_name=age_var,
                                       sas_variable_feature=feature_var)

if var_info:
    print(f"Informações para {age_var}:")
    print(json.dumps(var_info, indent=4, ensure_ascii=False))
else:
    print(f"Variável {age_var} não encontrada.")



# %% [markdown]
# ### 3.8 fração do Codebook Tabela

# %%


sas_variable_test = "_AGEG5YR"  # Substitua pelo nome da variável desejada
sas_variable_heart = "CVDSTRK3"  # Substitua pelo nome da variável desejada
# CVDINFR4 Heart Attack
# CVDCRHD4 Angina
# CVDSTRK3 Stroke


# Caminho do JSON gerado

json_path_file_to_csv = f"{notebook_dir_project_predict}data{os.sep}intermediate{os.sep}2023{os.sep}brfss_2023_variaveis.json"

var_info = consultar_var_info(json_file_path=json_path_file_to_csv, sas_variable_name=sas_variable_heart)

if var_info:
    # Criar DataFrame com os dados de var_info
    df_var_info = pd.DataFrame(var_info)
    
else:
    print(f"Variável {sas_variable_heart} não encontrada.")

df_var_info

# %% [markdown]
# CONTINUA EM EXPLORE


