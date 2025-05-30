import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Para mais customização se necessário
import mlflow
import os

# --- Suponha que df_processado seja seu DataFrame ---
# Exemplo de DataFrame
data_exemplo = {
    'IDADE': [25, 30, 30, 35, 40, 40, 40, 45, 50, 55, 60, 25, 30, 80, 90, 55, 28, 33, 42, 58],
    'GENERO': ['F', 'F', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M'],
    '_MICHD': [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
}
df_processado = pd.DataFrame(data_exemplo)

# --- Iniciar uma execução do MLflow ---
EXPERIMENT_NAME = "BRFSS_Risco_Cardiaco_Plotly"
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="Analise_Descritiva_Entrada_Plotly") as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # Criar uma pasta para salvar os gráficos HTML, se não existir
    graficos_plotly_path = "graficos_plotly_dataset"
    if not os.path.exists(graficos_plotly_path):
        os.makedirs(graficos_plotly_path)

    # --- Gráfico 1: Histograma Interativo de Idades ---
    if 'IDADE' in df_processado.columns:
        fig_hist_idade = px.histogram(df_processado.dropna(subset=['IDADE']), 
                                      x="IDADE", 
                                      nbins=20, 
                                      title="Distribuição Interativa de Idades",
                                      labels={'IDADE': 'Idade'})
        fig_hist_idade.update_layout(bargap=0.1) # Espaçamento entre as barras
        
        caminho_hist_idade_plotly = os.path.join(graficos_plotly_path, "distribuicao_idades_plotly.html")
        fig_hist_idade.write_html(caminho_hist_idade_plotly)
        print(f"Histograma Plotly de idades salvo em: {caminho_hist_idade_plotly}")
        
        mlflow.log_artifact(caminho_hist_idade_plotly, artifact_path="analise_plotly_entrada")
        print(f"Artefato '{caminho_hist_idade_plotly}' logado no MLflow.")

    # --- Gráfico 2: Box Plot de Idades por Gênero (Exemplo) ---
    if 'IDADE' in df_processado.columns and 'GENERO' in df_processado.columns:
        fig_boxplot_idade_genero = px.box(df_processado.dropna(subset=['IDADE', 'GENERO']), 
                                          x="GENERO", 
                                          y="IDADE", 
                                          color="GENERO",
                                          title="Box Plot de Idades por Gênero",
                                          labels={'IDADE': 'Idade', 'GENERO': 'Gênero'})
        
        caminho_boxplot_plotly = os.path.join(graficos_plotly_path, "boxplot_idade_genero_plotly.html")
        fig_boxplot_idade_genero.write_html(caminho_boxplot_plotly)
        print(f"Box Plot Plotly salvo em: {caminho_boxplot_plotly}")

        mlflow.log_artifact(caminho_boxplot_plotly, artifact_path="analise_plotly_entrada")
        print(f"Artefato '{caminho_boxplot_plotly}' logado no MLflow.")

    # Você pode adicionar mais gráficos aqui (ex: contagem de _MICHD, etc.)

    # ... (resto do seu código de treinamento, log de métricas, modelo, etc.) ...

    print("Execução do MLflow com gráficos Plotly concluída.")