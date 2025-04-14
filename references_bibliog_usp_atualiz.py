import re
'''
def read_nbib(nbib_file):
    """Lê o arquivo .nbib como texto."""
    with open(nbib_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def parse_nbib(content):
    """Extrai e formata as referências do conteúdo do arquivo .nbib no padrão ABNT especificado."""
    entries = content.strip().split("\n\n")  # Divide cada entrada
    references = []

    for entry in entries:
        # Extrai os campos principais com tratamento de erro
        title = re.search(r'(?<=TI  - ).*', entry)
        author = re.findall(r'(?<=AU  - ).*', entry)
        journal = re.search(r'(?<=JT  - ).*', entry)
        year = re.search(r'(?<=DP  - )\d{4}', entry)
        volume = re.search(r'(?<=VI  - )\d+', entry)
        issue = re.search(r'(?<=IP  - )\d+', entry)
        
        # Captura página única (PG  - 126) ou faixa (PG  - 126-140)
        pages = re.search(r'(?<=PG  - )(\d+)(?:-(\d+))?', entry)

        # Define valores padrão para evitar erro de 'NoneType'
        title = title.group() if title else "Título desconhecido"
        year = year.group() if year else "????"
        journal = journal.group() if journal else "Revista desconhecida"
        volume = volume.group() if volume else "?"
        issue = issue.group() if issue else "?"
        
        # Ajusta a formatação das páginas
        if pages:
            if pages.group(2):  # Se houver uma faixa de páginas
                pages_formatted = f"{pages.group(1)}-{pages.group(2)}"
            else:  # Se houver apenas uma página
                pages_formatted = pages.group(1)
        else:
            pages_formatted = "??"

        authors = "; ".join(author) if author else "Autor desconhecido"

        # Monta a referência no padrão solicitado
        ref = f"{authors}. {year}. {title}. {journal} {volume}({issue}): {pages_formatted}."

        references.append(ref)

    return references

# Exemplo de uso
nbib_file = "/home/ed/lgcm/projects/riskPredictionDeseases/citations-20250313T185936.ris"
content = read_nbib(nbib_file)  # Lê o conteúdo do arquivo
refs = parse_nbib(content)  # Extrai as referências

# Salvar em um arquivo .txt
with open("referencias_abnt_usp_esalq.txt", "w", encoding="utf-8") as f:
    for ref in refs:
        f.write(ref + "\n\n")

print("Referências formatadas salvas em 'referencias_abnt_usp_esalq.txt'.")

###################################################################################

import re

def read_file(file_path):
    """Lê um arquivo .nbib ou .ris como texto."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def parse_references(content, file_type):
    """Extrai e formata referências de arquivos .nbib e .ris no padrão ABNT."""
    entries = content.strip().split("\n\n")  # Divide cada entrada
    references = []

    for entry in entries:
        if file_type == "nbib":
            # Extrai dados do formato NBIB
            title = re.search(r'(?<=TI  - ).*', entry)
            author = re.findall(r'(?<=AU  - ).*', entry)
            journal = re.search(r'(?<=JT  - ).*', entry)
            year = re.search(r'(?<=DP  - )\d{4}', entry)
            volume = re.search(r'(?<=VI  - )\d+', entry)
            issue = re.search(r'(?<=IP  - )\d+', entry)
            pages = re.search(r'(?<=PG  - )(\d+)(?:-(\d+))?', entry)  # Captura página única ou faixa

        elif file_type == "ris":
            # Extrai dados do formato RIS
            title = re.search(r'(?<=T1  - ).*', entry)
            author = re.findall(r'(?<=AU  - ).*', entry)
            journal = re.search(r'(?<=JO  - ).*', entry)  # JO representa Journal em RIS
            year = re.search(r'(?<=PY  - )\d{4}', entry)  # PY representa Ano em RIS
            volume = re.search(r'(?<=VL  - )\d+', entry)
            issue = re.search(r'(?<=IS  - )\d+', entry)  # IS representa Número em RIS
            pages = re.search(r'(?<=SP  - )(\d+)(?:-(\d+))?', entry)  # SP representa Página Inicial

        else:
            continue  # Se o tipo do arquivo não for reconhecido, pula a entrada

        # Define valores padrão para evitar erro de 'NoneType'
        title = title.group() if title else "Título desconhecido"
        year = year.group() if year else "????"
        journal = journal.group() if journal else "Revista desconhecida"
        volume = volume.group() if volume else "?"
        issue = issue.group() if issue else "?"

        # Ajusta a formatação das páginas
        if pages:
            pages_formatted = f"{pages.group(1)}-{pages.group(2)}" if pages.group(2) else pages.group(1)
        else:
            pages_formatted = "??"

        authors = "; ".join(author) if author else "Autor desconhecido"

        # Monta a referência no padrão solicitado
        ref = f"{authors}. {year}. {title}. {journal} {volume}({issue}): {pages_formatted}."
        references.append(ref)

    return references

# Exemplo de uso
# varrer o path myDir/references
# e pegar todos os arquivos .ris e .nbib que estão nas subpastas
# e iterar neles gerando as referencias 



file_path = "/home/ed/lgcm/projects/riskPredictionDeseases/citations-20250313T185936.ris"  # Ou "referencias.ris"
file_type = "nbib" if file_path.endswith(".nbib") else "ris"

content = read_file(file_path)  # Lê o conteúdo do arquivo
refs = parse_references(content, file_type)  # Extrai as referências

# Salvar em um arquivo .txt
with open("referencias_abnt.txt", "w", encoding="utf-8") as f:
    for ref in refs:
        f.write(ref + "\n\n")

print("Referências formatadas salvas em 'referencias_abnt.txt'.")
'''
import os
import re

def read_file(file_path):
    """Lê um arquivo .nbib ou .ris como texto."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def parse_references(content, file_type):
    """Extrai e formata referências de arquivos .nbib e .ris no padrão ABNT."""
    entries = content.strip().split("\n\n")  # Divide cada entrada
    references = []

    for entry in entries:
        if file_type == "nbib":
            # Extrai dados do formato NBIB
            title = re.search(r'(?<=TI  - ).*', entry)
            author = re.findall(r'(?<=AU  - ).*', entry)
            journal = re.search(r'(?<=JT  - ).*', entry)
            year = re.search(r'(?<=DP  - )\d{4}', entry)
            volume = re.search(r'(?<=VI  - )\d+', entry)
            issue = re.search(r'(?<=IP  - )\d+', entry)
            pages = re.search(r'(?<=PG  - )(\d+)(?:-(\d+))?', entry)  # Captura página única ou faixa

        elif file_type == "ris":
            # Extrai dados do formato RIS
            title = re.search(r'(?<=T1  - ).*', entry)
            author = re.findall(r'(?<=AU  - ).*', entry)
            journal = re.search(r'(?<=JO  - ).*', entry)  # JO representa Journal em RIS
            year = re.search(r'(?<=PY  - )\d{4}', entry)  # PY representa Ano em RIS
            volume = re.search(r'(?<=VL  - )\d+', entry)
            issue = re.search(r'(?<=IS  - )\d+', entry)  # IS representa Número em RIS
            pages = re.search(r'(?<=SP  - )(\d+)(?:-(\d+))?', entry)  # SP representa Página Inicial

        else:
            continue  # Se o tipo do arquivo não for reconhecido, pula a entrada

        # Define valores padrão para evitar erro de 'NoneType'
        title = title.group() if title else "Título desconhecido"
        year = year.group() if year else "????"
        journal = journal.group() if journal else "Revista desconhecida"
        volume = volume.group() if volume else "?"
        issue = issue.group() if issue else "?"

        # Ajusta a formatação das páginas
        if pages:
            pages_formatted = f"{pages.group(1)}-{pages.group(2)}" if pages.group(2) else pages.group(1)
        else:
            pages_formatted = "??"

        authors = "; ".join(author) if author else "Autor desconhecido"

        # Monta a referência no padrão solicitado
        ref = f"{authors}. {year}. {title}. {journal} {volume}({issue}): {pages_formatted}."
        references.append(ref)

    return references

def find_files(directory):
    """Encontra todos os arquivos .ris e .nbib no diretório e subdiretórios."""
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ris") or file.endswith(".nbib"):
                file_paths.append(os.path.join(root, file))
    return file_paths

# Caminho do diretório a ser varrido
directory = "/home/ed/lgcm/projects/riskPredictionDeseases/data/input/references"
file_paths = find_files(directory)

# Lista para armazenar todas as referências
all_references = []

# Iterar sobre os arquivos encontrados e extrair as referências
for file_path in file_paths:
    file_type = "nbib" if file_path.endswith(".nbib") else "ris"
    content = read_file(file_path)
    references = parse_references(content, file_type)
    all_references.extend(references)

# Salvar todas as referências em um único arquivo .txt
with open("referencias_abnt_march_2025.txt", "w", encoding="utf-8") as f:
    for ref in all_references:
        f.write(ref + "\n\n")

print("Referências formatadas salvas em 'referencias_abnt_march_2025.txt'.")