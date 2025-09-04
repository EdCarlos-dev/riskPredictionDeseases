import re

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
        pages = re.search(r'(?<=PG  - )(\d+)-(\d+)', entry)  # Captura a faixa de páginas

        # Define valores padrão para evitar erro de 'NoneType'
        title = title.group() if title else "Título desconhecido"
        year = year.group() if year else "????"
        journal = journal.group() if journal else "Revista desconhecida"
        volume = volume.group() if volume else "?"
        issue = issue.group() if issue else "?"
        pages_formatted = f"{pages.group(1)}-{pages.group(2)}" if pages else "??-??"

        authors = "; ".join(author) if author else "Autor desconhecido"

        # Monta a referência no padrão solicitado
        ref = f"{authors}. {year}. {title}. {journal} {volume}({issue}): {pages_formatted}."

        references.append(ref)

    return references

# Exemplo de uso
nbib_file = "pmc_11538124.nbib"
content = read_nbib(nbib_file)  # Lê o conteúdo do arquivo
refs = parse_nbib(content)  # Extrai as referências

# Salvar em um arquivo .txt
with open("referencias_abnt_usp.txt", "w", encoding="utf-8") as f:
    for ref in refs:
        f.write(ref + "\n\n")

print("Referências formatadas salvas em 'referencias_abnt_usp.txt'.")
