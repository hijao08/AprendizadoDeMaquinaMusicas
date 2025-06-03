#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análise Exploratória do Dataset "Top 100 Songs & Lyrics By Year (1959-2023)"
Foco: Detecção de conteúdo inapropriado (racismo, homofobia, discurso de ódio, preconceito)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import warnings
import os
import sys

# Configurações iniciais
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')  # Versão específica do estilo seaborn
sns.set_theme()  # Usando o tema padrão do seaborn

# Configuração de diretórios
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'figures')

# Criar diretório de resultados se não existir
os.makedirs(RESULTS_DIR, exist_ok=True)

def setup_nltk():
    """Configura os recursos necessários do NLTK"""
    try:
        # Download de todos os recursos necessários
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Erro ao baixar recursos do NLTK: {e}")
        sys.exit(1)

# Lista de termos sensíveis para análise
TERMOS_SENSIVEIS = {
   'racismo': [
        'nigger', 'nigga', 'coon', 'spic', 'wetback', 'chink', 'gook', 'kike', 'spook', 'wop',
        'jungle bunny', 'porch monkey', 'sambo', 'pickaninny', 'tar baby',
        'redskin', 'towelhead', 'sand nigger', 'camel jockey', 'oreo',
        # novos termos antigos ou pouco mapeados
        'darkie', 'half-breed', 'injun', 'boy' , 'yellow', 'mulatto', 'negroid', 'ape', 
        'colored', 'negro', 'monkey', 'spear chucker', 'uncle tom'
    ],
    'homofobia': [
        'fag', 'faggot', 'dyke', 'queer', 'homo', 'fairy', 'butch', 'tranny',
        'no homo', 'sissy', 'pillow biter', 'bender', 'fruitcake', 'fudge packer',
        'he-she', 'ladyboy', 'gender bender', 'nancy boy', 'manlover', 'fenboy', 
        'fagboy', 'fagboyz',
        # adicionais antigos
        'limp wrist', 'light in the loafers', 'pansy', 'poof', 'poofter', 'nelly',
        'flamer', 'queen', 'switch hitter', 'shemale', 'invert'
    ],
    'discurso_odio': [
        'hate', 'kill', 'death', 'murder', 'blood', 'war', 'fight', 'enemy',
        'shoot', 'stab', 'explode', 'terrorist', 'bomb', 'massacre',
        'annihilate', 'burn', 'genocide', 'lynch', 'torture', 'rape',
        # adicionais comuns em músicas antigas
        'riot', 'slaughter', 'execute', 'hang', 'pistol', 'bullet',
        'rage', 'beatdown', 'warzone', 'snipe', 'trigger', 'reload', 'strangle'
    ],
    'preconceito': [
        'stupid', 'dumb', 'ugly', 'fat', 'lazy', 'crazy', 'retard', 'whore', 'slut',
        'bitch', 'hoe', 'loser', 'idiot', 'moron', 'psycho', 'bastard',
        'cunt', 'skank', 'airhead', 'gold digger', 'pig', 'slob',
        'twat', 'tramp', 'hag', 'freak', 'cripple', 'lame',
        # adicionais de décadas passadas
        'bimbo', 'nutcase', 'loony', 'simpleton', 'ditz', 'clown', 'trollop',
        'nitwit', 'mutt', 'cow', 'dog', 'fatso', 'minger', 'uggo', 'whaleboy'
    ]
}

def carregar_dataset():
    """Carrega o dataset e realiza limpeza inicial"""
    print("Carregando o dataset...")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'all_songs_data.csv'))
        if df.empty:
            raise ValueError("O dataset está vazio")
        
        # Verifica se as colunas necessárias existem
        colunas_necessarias = ['Year', 'Lyrics', 'Song Title', 'Artist']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        if colunas_faltantes:
            raise ValueError(f"Colunas necessárias não encontradas: {colunas_faltantes}")
        
        # Converte a coluna Year para inteiro
        df['Year'] = df['Year'].astype(int)
        
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {os.path.join(DATA_DIR, 'all_songs_data.csv')}")
        print("Certifique-se de que o arquivo está na pasta 'data/'")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        sys.exit(1)

def analise_basica(df):
    """Realiza análise básica do dataset"""
    try:
        print("\n=== Informações Básicas do Dataset ===")
        print("\nTipos de dados:")
        print(df.dtypes)
        
        print("\nValores nulos por coluna:")
        print(df.isnull().sum())
        
        print("\nValores duplicados:", df.duplicated().sum())
        
        print("\nEstatísticas descritivas:")
        print(df.describe())
        
        print("\nPrimeiras linhas do dataset:")
        print(df.head())
    except Exception as e:
        print(f"Erro na análise básica: {e}")

def analise_temporal(df):
    """Analisa a distribuição temporal das músicas"""
    try:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='Year', bins=64)
        plt.title('Distribuição de Músicas por Ano (1959-2023)')
        plt.xlabel('Ano')
        plt.ylabel('Quantidade de Músicas')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'distribuicao_temporal.png'))
        plt.close()
    except Exception as e:
        print(f"Erro na análise temporal: {e}")

def analise_tamanho_letras(df):
    """Analisa o tamanho das letras das músicas"""
    try:
        # Calcula o número de palavras por letra
        df['num_palavras'] = df['Lyrics'].str.split().str.len()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='num_palavras', bins=50)
        plt.title('Distribuição do Tamanho das Letras (Número de Palavras)')
        plt.xlabel('Número de Palavras')
        plt.ylabel('Frequência')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'tamanho_letras.png'))
        plt.close()
    except Exception as e:
        print(f"Erro na análise do tamanho das letras: {e}")
        

def conta_termos(texto, termos):
    """Conta quantos termos aparecem em um texto (forma precisa usando regex com bordas de palavra)."""
    texto = str(texto).lower()
    return sum(1 for termo in termos if re.search(rf'\b{re.escape(termo)}\b', texto))


def analise_termos_sensiveis(df):
    """Analisa a frequência de termos sensíveis ao longo dos anos e plota sua evolução"""
    try:
        # Verifica se diretório de resultados existe
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Prepara a coluna com letras minúsculas
        df['lyrics_lower'] = df['Lyrics'].str.lower()
        
        # Dicionário para armazenar resultados por categoria
        resultados = {}

        for categoria, termos in TERMOS_SENSIVEIS.items():
            # Aplica a função para contar termos sensíveis por linha
            df[f'freq_{categoria}'] = df['lyrics_lower'].apply(lambda x: conta_termos(x, termos))
            
            # Calcula a média anual de frequência dos termos
            media_anual = df.groupby('Year')[f'freq_{categoria}'].mean()
            resultados[categoria] = media_anual

        # Plotagem
        plt.figure(figsize=(15, 8))
        for categoria, dados in resultados.items():
            plt.plot(dados.index, dados.values, label=categoria.capitalize(), marker='o')

        plt.title('Evolução da Frequência de Termos Sensíveis (1959–2023)')
        plt.xlabel('Ano')
        plt.ylabel('Frequência Média por Letra')
        plt.legend(title="Categoria")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'evolucao_termos_sensiveis.png'))
        plt.close()

        print("Análise de termos sensíveis concluída com sucesso!")

    except Exception as e:
        print(f"Erro na análise de termos sensíveis: {e}")
        import traceback
        traceback.print_exc()

def analise_palavras_frequentes(df):
    """Analisa as palavras mais frequentes nas letras"""
    try:
        # Tokenização e limpeza
        stop_words = set(stopwords.words('english'))
        todas_palavras = []
        
        # Adiciona palavras comuns em músicas aos stopwords
        stop_words.update(['oh', 'yeah', 'hey', 'la', 'na', 'da', 'woah', 'ooh', 'ah', 'ha'])
        
        for letra in df['Lyrics'].dropna():
            # Tokenização simples usando split
            palavras = str(letra).lower().split()
            # Filtra palavras
            palavras = [p for p in palavras if p.isalpha() and p not in stop_words and len(p) > 2]
            todas_palavras.extend(palavras)
        
        # Contagem de frequência
        contador = Counter(todas_palavras)
        palavras_mais_frequentes = contador.most_common(20)
        
        # Plotagem
        plt.figure(figsize=(12, 6))
        palavras, frequencias = zip(*palavras_mais_frequentes)
        plt.barh(palavras, frequencias)
        plt.title('20 Palavras Mais Frequentes nas Letras')
        plt.xlabel('Frequência')
        plt.ylabel('Palavra')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'palavras_frequentes.png'))
        plt.close()
        
        # Imprime as palavras mais frequentes
        print("\n=== Palavras Mais Frequentes ===")
        for palavra, freq in palavras_mais_frequentes:
            print(f"{palavra}: {freq}")
            
    except Exception as e:
        print(f"Erro na análise de palavras frequentes: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Função principal"""
    print("Iniciando análise exploratória...")
    
    # Configura o NLTK
    setup_nltk()
    
    # Carrega o dataset
    df = carregar_dataset()
    
    # Realiza as análises
    analise_basica(df)
    analise_temporal(df)
    analise_tamanho_letras(df)
    analise_termos_sensiveis(df)
    analise_palavras_frequentes(df)
    
    print("\nAnálise concluída! Os gráficos foram salvos na pasta 'results/figures'.")

if __name__ == "__main__":
    main() 