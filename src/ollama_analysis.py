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

    def add_exemplo_manual(self, letra: str, score: str, justificativa: str):
        """Adiciona um exemplo de classificação manual para aprendizado"""
        self.exemplos_manuais.append({
            "letra": letra,
            "score": score,
            "justificativa": justificativa
        })

    def gerar_prompt_com_exemplos(self, prompt_base: str) -> str:
        """Gera o prompt completo incluindo os exemplos de classificação manual"""
        prompt_completo = prompt_base + "\n\n=== EXEMPLOS DE REFERÊNCIA ===\n"
        prompt_completo += "Analise cuidadosamente os seguintes exemplos já classificados. Eles devem servir como base para suas próximas classificações.\n"
        
        for exemplo in self.exemplos_manuais[:5]:
            prompt_completo += "\n---\nEXEMPLO DE CLASSIFICAÇÃO:\n"
            prompt_completo += f"Letra:\n{exemplo['letra'][:800]}...\n\n"  # Aumentei o limite de caracteres
            prompt_completo += f"Nivel de toxicidade: {exemplo['score']}\n"
            prompt_completo += f"Justificativa: {exemplo['justificativa']}\n"
            prompt_completo += "---\n"
        
        prompt_completo += "\nAgora, use esses exemplos como referência para classificar a próxima música de forma similar.\n"
        return prompt_completo

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Tenta converter a resposta do modelo em um dicionário estruturado"""
        try:
            # Primeiro tenta parsear como JSON direto
            return json.loads(response)
        except json.JSONDecodeError:
            # Se falhar, tenta extrair os campos do texto
            result = {
                "nivel_toxicidade": "NA",
                "abuso_emocional": False,
                "ciume_possessividade": False,
                "dependencia": False,
                "objetificacao": False,
                "violencia_traicao": False,
                "justificativa": ""
            }
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip().lower()
                if line.startswith('nivel de toxicidade:'):
                    nivel = line.split(':')[1].strip()
                    if 'muito baixo' in nivel:
                        result['nivel_toxicidade'] = 'muito baixo'
                    elif 'baixo' in nivel:
                        result['nivel_toxicidade'] = 'baixo'
                    elif 'moderado' in nivel:
                        result['nivel_toxicidade'] = 'moderado'
                    elif 'alto' in nivel:
                        result['nivel_toxicidade'] = 'alto'
                    elif 'muito alto' in nivel:
                        result['nivel_toxicidade'] = 'muito alto'
                    else:
                        result['nivel_toxicidade'] = 'na'
                elif line.startswith('abuso_emocional:'):
                    result['abuso_emocional'] = 'yes' in line.lower() or 'true' in line.lower()
                elif line.startswith('ciume_possessividade:'):
                    result['ciume_possessividade'] = 'yes' in line.lower() or 'true' in line.lower()
                elif line.startswith('dependencia:'):
                    result['dependencia'] = 'yes' in line.lower() or 'true' in line.lower()
                elif line.startswith('objetificacao:'):
                    result['objetificacao'] = 'yes' in line.lower() or 'true' in line.lower()
                elif line.startswith('violencia_traicao:'):
                    result['violencia_traicao'] = 'yes' in line.lower() or 'true' in line.lower()
                elif line.startswith('justificativa:'):
                    result['justificativa'] = line.split(':', 1)[1].strip()
            
            return result

    def analyze_text(self, text: str, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            if len(text) > 2000:
                text = text[:2000]

            # Sempre usa os exemplos, não mais condicional
            full_prompt = f"{self.gerar_prompt_com_exemplos(prompt)}\n\nAGORA ANALISE ESTA LETRA:\n{text}"

            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    'temperature': 0.1,  # Mantém baixo para consistência
                    'num_ctx': 4096,     # Aumentei o contexto para caber mais exemplos
                    'num_predict': 1000   # Aumentei para respostas mais detalhadas
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
                nivel = row['Pontuacao_manual'] if 'Pontuacao_manual' in df_manual.columns else None
                justificativa = row['Justificativa'] if 'Justificativa' in df_manual.columns else None
                
                if letra is not None and nivel is not None:
                    analyzer.add_exemplo_manual(
                        letra=letra,
                        score=nivel,
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
    Você é um especialista em análise crítica de letras de músicas, focado em identificar elementos tóxicos em relacionamentos amorosos.

    1. OBJETIVO:
    Analisar letras de músicas e avaliar se elas contêm temas que incitam, normalizam ou validam comportamentos tóxicos em relacionamentos.

    2. NÍVEIS DE TOXICIDADE:
    [muito alto] - Múltiplos elementos tóxicos graves
    [alto] - Elementos tóxicos significativos
    [moderado] - Alguns elementos problemáticos
    [baixo] - Poucos elementos questionáveis
    [muito baixo] - Elementos mínimos ou ambíguos
    [na] - Sem elementos tóxicos identificáveis

    3. ELEMENTOS A IDENTIFICAR:
    • Dependência Emocional: necessidade excessiva do outro, incapacidade de ficar sozinho
    • Ciúme/Possessividade: controle, posse, restrição da liberdade do outro
    • Violência: física, psicológica, ameaças, intimidação
    • Manipulação/Abuso Emocional: gaslighting, chantagem, culpabilização
    • Objetificação: redução da pessoa a objeto sexual ou de posse
    • Submissão Tóxica: anulação de si, aceitação de maus-tratos

    4. FORMATO DA RESPOSTA:
    Nivel de toxicidade: [na/muito baixo/baixo/moderado/alto/muito alto]
    abuso_emocional: [yes/no]
    ciume_possessividade: [yes/no]
    dependencia: [yes/no]
    objetificacao: [yes/no]
    violencia_traicao: [yes/no]
    justificativa: [Cite trechos específicos da letra e explique como eles se relacionam com cada elemento tóxico identificado]

    5. REGRAS IMPORTANTES:
    • SEMPRE cite trechos específicos da letra para justificar sua classificação
    • Se não encontrar NENHUM dos elementos tóxicos listados, classifique como "na"
    • Não penalize expressões normais de amor, saudade ou desejo
    • Considere o contexto geral da letra, não apenas frases isoladas
    • Para práticas BDSM/submissão, só considere tóxico se claramente não consensual
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
                    "nivel_toxicidade": result.get("nivel_toxicidade", "NA"),
                    "abuso_emocional": result.get("abuso_emocional", False),
                    "ciume_possessividade": result.get("ciume_possessividade", False),
                    "dependencia": result.get("dependencia", False),
                    "objetificacao": result.get("objetificacao", False),
                    "violencia_traicao": result.get("violencia_traicao", False),
                    "justificativa": result.get("justificativa", "")
                })

                print(f"→ Nível de toxicidade: {result.get('nivel_toxicidade', 'NA')}")

                # Salva a cada 5 análises
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
        print("\n📊 Carregando dataset de músicas classificadas...")
        df_manual = pd.read_csv("data/30-musicas-Mozart.csv")
        
        print("\n📊 Carregando dataset de músicas para análise...")
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
    
    # Contagem por nível de toxicidade
    niveis = resultados_df['nivel_toxicidade'].value_counts()
    print("\nDistribuição por nível de toxicidade:")
    for nivel, count in niveis.items():
        print(f"→ {nivel}: {count} músicas ({(count/len(resultados_df)*100):.1f}%)")
    
    # Contagem de elementos tóxicos
    print("\nPrevalência de elementos tóxicos:")
    print(f"→ Abuso emocional: {resultados_df['abuso_emocional'].sum()} músicas")
    print(f"→ Ciúme/possessividade: {resultados_df['ciume_possessividade'].sum()} músicas")
    print(f"→ Dependência: {resultados_df['dependencia'].sum()} músicas")
    print(f"→ Objetificação: {resultados_df['objetificacao'].sum()} músicas")
    print(f"→ Violência/traição: {resultados_df['violencia_traicao'].sum()} músicas")

if __name__ == "__main__":
    main()
