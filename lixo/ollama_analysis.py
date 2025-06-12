import pandas as pd
import ollama
import json
from typing import Dict, Any, Optional
import os
from datetime import datetime

class OllamaAnalyzer:
    def __init__(self, model: str = "mistral:instruct"):
        self.model = model
        self.exemplos_manuais = []

    def add_exemplo_manual(self, letra: str, score: float, justificativa: str):
        """Adiciona um exemplo de classificação manual para aprendizado"""
        self.exemplos_manuais.append({
            "letra": letra,
            "score": score,
            "justificativa": justificativa
        })

    def gerar_prompt_com_exemplos(self, prompt_base: str) -> str:
        """Gera o prompt completo incluindo os exemplos de classificação manual"""
        prompt_completo = prompt_base + "\n\n**Exemplos de classificação:**\n"
        
        for exemplo in self.exemplos_manuais[:3]:  # Limitamos a 3 exemplos para não sobrecarregar
            prompt_completo += f"\nLetra:\n{exemplo['letra'][:500]}...\n"  # Limitamos o tamanho da letra
            prompt_completo += f"Score: {exemplo['score']}\n"
            prompt_completo += f"Justificativa: {exemplo['justificativa']}\n"
            prompt_completo += "-" * 50 + "\n"
        
        return prompt_completo

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Tenta converter a resposta do modelo em um dicionário estruturado"""
        try:
            # Primeiro tenta parsear como JSON direto
            return json.loads(response)
        except json.JSONDecodeError:
            # Se falhar, tenta extrair os campos do texto
            result = {
                "score": 0.0,
                "emotional_abuse": False,
                "jealousy_possessive": False,
                "dependency": False,
                "objectification": False,
                "violence_or_betrayal": False,
                "justification": ""
            }
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip().lower()
                if line.startswith('score:'):
                    try:
                        result['score'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('emotional_abuse:'):
                    result['emotional_abuse'] = 'yes' in line.lower() or 'true' in line.lower()
                elif line.startswith('jealousy_possessive:'):
                    result['jealousy_possessive'] = 'yes' in line.lower() or 'true' in line.lower()
                elif line.startswith('dependency:'):
                    result['dependency'] = 'yes' in line.lower() or 'true' in line.lower()
                elif line.startswith('objectification:'):
                    result['objectification'] = 'yes' in line.lower() or 'true' in line.lower()
                elif line.startswith('violence_or_betrayal:'):
                    result['violence_or_betrayal'] = 'yes' in line.lower() or 'true' in line.lower()
                elif line.startswith('justificativa:') or line.startswith('justification:'):
                    result['justification'] = line.split(':', 1)[1].strip()
            
            return result

    def analyze_text(self, text: str, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            if len(text) > 2000:
                text = text[:2000]

            # Gera o prompt completo com exemplos se houver exemplos manuais
            if self.exemplos_manuais:
                full_prompt = f"{self.gerar_prompt_com_exemplos(prompt)}\n\nAgora analise esta letra:\n{text}"
            else:
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
            
            if response:
                return self.parse_response(response['response'].strip())
            return None
            
        except Exception as e:
            print(f"Erro ao analisar texto: {e}")
            return None

def carregar_exemplos_manuais(analyzer: OllamaAnalyzer, csv_path: str):
    """Carrega os exemplos de classificação manual para o analisador"""
    try:
        df_manual = pd.read_csv(csv_path)
        print(f"\nCarregando {len(df_manual)} exemplos de classificação manual...")
        
        for idx, row in df_manual.iterrows():
            try:
                letra = row['Letra'] if 'Letra' in df_manual.columns else None
                score = row['Pontuacao_manual'] if 'Pontuacao_manual' in df_manual.columns else None
                justificativa = row['Justificativa'] if 'Justificativa' in df_manual.columns else None
                
                if letra is not None and score is not None:
                    analyzer.add_exemplo_manual(
                        letra=letra,
                        score=float(score),
                        justificativa=str(justificativa) if justificativa is not None else ""
                    )
            except Exception as e:
                print(f"⚠️ Erro ao processar exemplo {idx}: {e}")
                continue
        
        print("✅ Exemplos manuais carregados com sucesso!")
    except Exception as e:
        print(f"⚠️ Erro ao carregar exemplos manuais: {e}")

def analise_conteudo_toxico(df: pd.DataFrame) -> pd.DataFrame:
    print("\nColunas disponíveis no dataset:")
    print(df.columns.tolist())

    coluna_titulo = 'Song Title'
    coluna_artista = 'Artist'
    coluna_letra = 'Lyrics'
    coluna_ano = 'Year'

    prompt = """
    Você é um especialista em análise crítica de letras de músicas. 
    Avalie se letras em inglês contêm temas que incitam, 
    normalizam ou validam comportamentos tóxicos em relacionamentos,
    aprenda com o exemplo das musicas que classifiquei manualmente e classifique as musicas conforme os critérios abaixo:

    **Critérios:**
    - Normalização de controle, ciúmes ou possessividade como sinais de amor.
    - Romantização de abuso no contexto de relacionamentos.
    - Incentivo a permanecer em relações prejudiciais por amor, destino ou paixão.
    - Reforço de papéis de gênero rígidos, submissão ou dominação no contexto de relacionamentos.
    - Narrativas que culpam a vítima ou desconsideram consentimento no contexto de relacionamentos.
    - Glorificação de dependência emocional/financeira como prova de amor no contexto de relacionamentos.
    - Linguagem que retrata obsessão como paixão ("I can't live without you") no contexto de relacionamentos.
    - Justifique com trechos de amor e abuso no contexto de relacionamentos.

    **Instruções:**
    - Caso a letra não seja sobre relacionamentos, retorne score 0.0.
    - O Score é uma escala de 0.0 a 1.0.
    - Só atribua score > 0.0 se houver clara evidência de um dos critérios acima.
    - Não penalize expressões comuns de afeto, desejo ou solidão, exceto se associadas a manipulação, controle ou submissão.
    - Cite sempre o trecho da letra que motivou o score e relacione ao critério.
    - Ignore ambiguidades não relacionadas a toxicidade relacional ou gênero.
    - Use os campos: score, emotional_abuse, jealousy_possessive, dependency, objectification, violence_or_betrayal, justificativa.

    Responda APENAS com esses campos, um por linha, sem texto adicional:
    score: [número entre 0.0 e 1.0]
    emotional_abuse: [yes/no]
    jealousy_possessive: [yes/no]
    dependency: [yes/no]
    objectification: [yes/no]
    violence_or_betrayal: [yes/no]
    justificativa: [explicação detalhada citando trechos]
    """

    analyzer = OllamaAnalyzer()
    
    # Carrega exemplos de classificação manual
    carregar_exemplos_manuais(analyzer, 'data/30-musicas-Mozart.csv')

    resultados = []
    os.makedirs("results", exist_ok=True)

    total = len(df)
    for idx, row in df.iterrows():
        print(f"\nAnalisando {idx + 1}/{total} - {row[coluna_titulo]}")

        result = analyzer.analyze_text(row[coluna_letra], prompt)
        if result:
            try:
                resultados.append({
                    "indice": idx,
                    "titulo": row[coluna_titulo],
                    "artista": row[coluna_artista],
                    "ano": row[coluna_ano] if coluna_ano in df.columns else None,
                    "score": result.get("score", 0.0),
                    "emotional_abuse": result.get("emotional_abuse", False),
                    "jealousy_possessive": result.get("jealousy_possessive", False),
                    "dependency": result.get("dependency", False),
                    "objectification": result.get("objectification", False),
                    "violence_or_betrayal": result.get("violence_or_betrayal", False),
                    "justificativa": result.get("justification", "")
                })

                print(f"→ Score: {result.get('score', 0)}")

                # Salva a cada 50 análises
                if (idx + 1) % 5 == 0:
                    parcial_path = f"results/parcial_{idx + 1}.csv"
                    pd.DataFrame(resultados).to_csv(parcial_path, index=False)
                    print(f"[Salvo parcial em: {parcial_path}]")

            except Exception as e:
                print(f"Erro ao processar resultado: {e}")
        else:
            print("Resposta nula do modelo.")

    return pd.DataFrame(resultados)

def main():
    try:
        print("\n📊 Carregando dataset...")
        df = pd.read_csv("data/30musicas.csv")
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
    print(f"→ Músicas com conteúdo tóxico (score > 0.5): {len(resultados_df[resultados_df['score'] > 0.5])}")

if __name__ == "__main__":
    main()
