import re

def read_nbib(nbib_file):
    """Lê o arquivo .nbib como texto."""
    with open(nbib_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def parse_nbib(content):
    """Extrai e formata as referências do conteúdo do arquivo .nbib."""
    entries = content.strip().split("\n\n")  # Divide cada entrada
    references = []

    for entry in entries:
        # Extrai os campos principais
        title = re.search(r'(?<=TI  - ).*', entry)
        author = re.findall(r'(?<=AU  - ).*', entry)
        journal = re.search(r'(?<=JT  - ).*', entry)
        year = re.search(r'(?<=DP  - )\d{4}', entry)
        volume = re.search(r'(?<=VI  - )\d+', entry)
        issue = re.search(r'(?<=IP  - )\d+', entry)
        pages = re.search(r'(?<=PG  - ).*', entry)
        doi = re.search(r'(?<=LID - )([^\s]+)(?=\s\[doi\])', entry)

        # Formata a referência no padrão ABNT
        if title and author and journal and year:
            authors = "; ".join(author) if author else "Autor desconhecido"
            ref = f"{authors}. {title.group()}. {journal.group()}, v. {volume.group() if volume else '?'}, n. {issue.group() if issue else '?'}, p. {pages.group() if pages else '?'}, {year.group()}. {f'DOI: {doi.group()}' if doi else ''}"
            references.append(ref)

    return references

# Exemplo de uso
nbib_file = "pmc_11538124.nbib"
content = read_nbib(nbib_file)  # Lê o conteúdo do arquivo
refs = parse_nbib(content)  # Extrai as referências

# Salvar em um arquivo .txt
with open("referencias_abnt.txt", "w", encoding="utf-8") as f:
    for ref in refs:
        f.write(ref + "\n\n")

print("Referências formatadas salvas em 'referencias_abnt.txt'.")
