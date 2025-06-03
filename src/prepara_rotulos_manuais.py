import pandas as pd
import numpy as np
import os

# Caminho do dataset original
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CSV_PATH = os.path.join(DATA_DIR, 'all_songs_data.csv')

# Caminho do arquivo de saída
OUTPUT_PATH = os.path.join(DATA_DIR, 'rotulos_manuais.csv')

# Número de músicas a serem selecionadas para rotulagem manual
N_AMOSTRA = 30

# 1. Carrega o dataset
# --------------------------------------------------
# Lê o arquivo CSV com as músicas
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)

# 2. Seleciona aleatoriamente N músicas
# --------------------------------------------------
# Remove músicas sem letra para garantir qualidade da amostra
# --------------------------------------------------
df_validas = df.dropna(subset=['Lyrics'])

# Seleciona aleatoriamente N músicas
amostra = df_validas.sample(n=N_AMOSTRA, random_state=42)

# 3. Prepara a estrutura para rotulagem manual
# --------------------------------------------------
# Cria um novo DataFrame apenas com os campos necessários e campos extras para anotação manual
# --------------------------------------------------
rotulos = pd.DataFrame({
    'Ano': amostra['Year'],
    'Artista': amostra['Artist'],
    'Título': amostra['Song Title'],
    'Letra': amostra['Lyrics'],
    'Pontuacao_manual': np.nan,  # Campo para ser preenchido manualmente (0 a 1)
    'Justificativa': '',         # Campo para justificativa manual
    'Referencia_criterio': ''    # (Opcional) Campo para referência ao critério
})

# 4. Salva a estrutura em CSV para preenchimento manual
# --------------------------------------------------
rotulos.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

# 5. Mensagem final
print(f"Arquivo para rotulagem manual salvo em: {OUTPUT_PATH}")
print("Preencha os campos 'Pontuacao_manual', 'Justificativa' e 'Referencia_criterio' conforme a avaliação da equipe.")

# -----------------------------
# Fim do script
# ----------------------------- 