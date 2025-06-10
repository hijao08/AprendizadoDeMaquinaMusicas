import pandas as pd
import ollama
import json
from typing import Dict, Any, Optional
import os
from datetime import datetime

class OllamaAnalyzer:
    def __init__(self, model: str = "mistral:instruct"):
        self.model = model

    def analyze_text(self, text: str, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            if len(text) > 2000:
                text = text[:2000]

            full_prompt = f"{prompt}\n\nLyrics:\n{text}"

            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    'temperature': 0.1,
                    'num_ctx': 2048,
                    'num_predict': 200
                }
            )
            return response
        except Exception as e:
            print(f"Erro ao analisar texto: {e}")
            return None

def analise_conteudo_toxico(df: pd.DataFrame) -> pd.DataFrame:
    print("\nColunas disponíveis no dataset:")
    print(df.columns.tolist())
    csv_classificado_manual = pd.read_csv('data/30-musicas-Mozart.csv')

    coluna_titulo = 'Song Title'
    coluna_artista = 'Artist'
    coluna_letra = 'Lyrics'
    coluna_ano = 'Year'

    prompt = """
    Você é um especialista em análise crítica de letras de músicas. Avalie se letras em inglês contêm temas que incitam, normalizam ou validam comportamentos tóxicos em relacionamentos, conforme os critérios abaixo:

    **Critérios:**
    - Normalização de controle, ciúmes ou possessividade como sinais de amor.
    - Romantização de abuso (gritos, manipulação, ameaças, humilhação).
    - Incentivo a permanecer em relações prejudiciais por amor, destino ou paixão.
    - Reforço de papéis de gênero rígidos, submissão ou dominação.
    - Narrativas que culpam a vítima ou desconsideram consentimento.
    - Glorificação de dependência emocional/financeira como prova de amor.
    - Linguagem que retrata obsessão como paixão (“I can’t live without you”).
    - Justifique com trechos de amor e abuso

    **Instruções:**
    - Só atribua score > 0.0 se houver clara evidência de um dos critérios acima.
    - Não penalize expressões comuns de afeto, desejo ou solidão, exceto se associadas a manipulação, controle ou submissão.
    - Cite sempre o trecho da letra que motivou o score e relacione ao critério.
    - Ignore ambiguidades não relacionadas a toxicidade relacional ou gênero.
    - Use os campos: score, emotional_abuse, jealousy_possessive, dependency, objectification, violence_or_betrayal, justificativa.
    """

    analyzer = OllamaAnalyzer()
    resultados = []

    os.makedirs("results", exist_ok=True)

    total = len(df)
    for idx, row in df.iterrows():
        print(f"\nAnalisando {idx + 1}/{total} - {row[coluna_titulo]}")

        result = analyzer.analyze_text(row[coluna_letra], prompt)
        if result:
            try:
                response = result['response'].strip()

                # Tenta carregar como JSON
                data = json.loads(response)

                resultados.append({
                    "indice": idx,
                    "titulo": row[coluna_titulo],
                    "artista": row[coluna_artista],
                    "ano": row[coluna_ano] if coluna_ano in df.columns else None,
                    "score": data.get("score", 0),
                    "emotional_abuse": data.get("emotional_abuse", False),
                    "jealousy_possessive": data.get("jealousy_possessive", False),
                    "dependency": data.get("dependency", False),
                    "violence_or_betrayal": data.get("violence_or_betrayal", False),
                    "justificativa": data.get("justification", "")
                })

                print(f"→ Score: {data.get('score', 0)}")

                # Salva a cada 50 análises
                if (idx + 1) % 50 == 0:
                    parcial_path = f"results/parcial_{idx + 1}.csv"
                    pd.DataFrame(resultados).to_csv(parcial_path, index=False)
                    print(f"[Salvo parcial em: {parcial_path}]")

            except Exception as e:
                print(f"Erro ao processar JSON: {e}")
                print(f"Resposta: {result['response']}")
        else:
            print("Resposta nula do modelo.")

    return pd.DataFrame(resultados)

def main():
    try:
        df = pd.read_csv("data/all_songs_data.csv")
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return

    print("\nAnalisando músicas para conteúdo tóxico em relacionamentos")
    print("=" * 60)
    print("Modelo em uso: mistral:instruct")
    print("=" * 60)

    resultados_df = analise_conteudo_toxico(df)

    if resultados_df.empty:
        print("Nenhum resultado retornado.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'results/relacionamentos_toxicos_{timestamp}.csv'
    json_path = f'results/relacionamentos_toxicos_{timestamp}.json'

    resultados_df.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(resultados_df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)

    print(f"\n✅ Análise completa!")
    print(f"Resultados salvos em:\n→ CSV: {csv_path}\n→ JSON: {json_path}")
    print(f"\nEstatísticas:")
    print(f"→ Total analisado: {len(resultados_df)}")
    print(f"→ Média score: {resultados_df['score'].mean():.2f}")
    print(f"→ Máximo: {resultados_df['score'].max():.2f}")
    print(f"→ Mínimo: {resultados_df['score'].min():.2f}")

if __name__ == "__main__":
    main()
