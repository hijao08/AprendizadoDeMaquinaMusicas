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

        for exemplo in self.exemplos_manuais:
            prompt_completo += "\n---\nEXEMPLO DE CLASSIFICAÇÃO:\n"
            prompt_completo += f"Letra:\n{exemplo['letra']}\n\n"
            prompt_completo += f"Nivel de toxicidade: {exemplo['score']}\n"
            prompt_completo += f"Justificativa: {exemplo['justificativa']}\n"
            prompt_completo += "---\n"
        
        prompt_completo += "\nAgora, use esses exemplos como referência para classificar a próxima música de forma similar.\n"
        return prompt_completo

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Tenta converter a resposta do modelo em um dicionário estruturado"""
        print(f"Resposta recebida: {response[:200]}...")  # Debug
        
        try:
            # Primeiro tenta parsear como JSON direto
            return json.loads(response)
        except json.JSONDecodeError:
            # Se falhar, tenta extrair os campos do texto
            result = {
                "nivel_toxicidade": "na",
                "abuso_emocional": False,
                "ciume_possessividade": False,
                "dependencia": False,
                "objetificacao": False,
                "violencia_traicao": False,
                "justificativa": ""
            }
            
            # Procura por JSON embeddado na resposta
            json_start = response.find('{')
            json_end = response.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                try:
                    json_part = response[json_start:json_end+1]
                    return json.loads(json_part)
                except json.JSONDecodeError:
                    pass
            
            # Parse manual mais robusto
            lines = response.split('\n')
            justificativa_lines = []
            capturing_justificativa = False
            
            for line in lines:
                line_clean = line.strip()
                line_lower = line_clean.lower()
                
                if line_lower.startswith('nivel de toxicidade:') or line_lower.startswith('nível de toxicidade:'):
                    nivel = line_clean.split(':', 1)[1].strip().lower()
                    if 'muito alto' in nivel:
                        result['nivel_toxicidade'] = 'muito alto'
                    elif 'muito baixo' in nivel:
                        result['nivel_toxicidade'] = 'muito baixo'
                    elif 'alto' in nivel:
                        result['nivel_toxicidade'] = 'alto'
                    elif 'baixo' in nivel:
                        result['nivel_toxicidade'] = 'baixo'
                    elif 'moderado' in nivel:
                        result['nivel_toxicidade'] = 'moderado'
                    elif 'na' in nivel or 'não aplicável' in nivel:
                        result['nivel_toxicidade'] = 'na'
                        
                elif line_lower.startswith('abuso_emocional:'):
                    val = line_clean.split(':', 1)[1].strip().lower()
                    result['abuso_emocional'] = 'true' in val or 'yes' in val or 'sim' in val
                    
                elif line_lower.startswith('ciume_possessividade:'):
                    val = line_clean.split(':', 1)[1].strip().lower()
                    result['ciume_possessividade'] = 'true' in val or 'yes' in val or 'sim' in val
                    
                elif line_lower.startswith('dependencia:') or line_lower.startswith('dependência:'):
                    val = line_clean.split(':', 1)[1].strip().lower()
                    result['dependencia'] = 'yes' in val or 'sim' in val or 'true' in val
                    
                elif line_lower.startswith('objetificacao:') or line_lower.startswith('objetificação:'):
                    val = line_clean.split(':', 1)[1].strip().lower()
                    result['objetificacao'] = 'yes' in val or 'sim' in val or 'true' in val
                    
                elif line_lower.startswith('violencia_traicao:') or line_lower.startswith('violência_traição:'):
                    val = line_clean.split(':', 1)[1].strip().lower()
                    result['violencia_traicao'] = 'yes' in val or 'sim' in val or 'true' in val
                    
                elif line_lower.startswith('justificativa:'):
                    capturing_justificativa = True
                    just_content = line_clean.split(':', 1)[1].strip()
                    if just_content:
                        justificativa_lines.append(just_content)
                        
                elif capturing_justificativa and line_clean:
                    justificativa_lines.append(line_clean)
            
            if justificativa_lines:
                result['justificativa'] = ' '.join(justificativa_lines)
                
            print(f"Resultado parseado: {result}")  # Debug
            return result

    def is_valid_response(self, result: Dict[str, Any]) -> bool:
        """Valida se a resposta contém os campos necessários e não está vazia"""
        if not result:
            return False
            
        # Verifica se tem nível de toxicidade válido
        valid_levels = ['na', 'muito baixo', 'baixo', 'moderado', 'alto', 'muito alto']
        if result.get('nivel_toxicidade', '').lower() not in valid_levels:
            return False
            
        # Se o nível não é 'na', deve ter justificativa
        if result.get('nivel_toxicidade', '').lower() != 'na' and not result.get('justificativa', '').strip():
            return False
            
        return True

    def analyze_text(self, text: str, prompt: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Analisa texto com retry automático em caso de resposta inválida"""
        for attempt in range(max_retries):
            try:
                print(f"Tentativa {attempt + 1}/{max_retries}")
                
                full_prompt = f"{self.gerar_prompt_com_exemplos(prompt)}\n\nAGORA ANALISE ESTA LETRA:\n{text}"

                response = ollama.generate(
                    model=self.model,
                    prompt=full_prompt,
                    options={
                        'temperature': 0.3,  # Consistência
                        'num_ctx': 8192,     # Contexto maior
                        'num_predict': 800,        # Redução para respostas mais focadas
                        'top_p': 0.7,              # Mais foco nas opções mais relevantes
                        'repeat_penalty': 1.2,     # Penalidade maior contra repetições
                        'top_k': 40,               # Novo parâmetro para limitar escolhas léxicas
                        'seed': 42                 # Reprodutibilidade
                    }
                )
                
                if response and response.get('response'):
                    result = self.parse_response(response['response'].strip())
                    
                    if self.is_valid_response(result):
                        print(f"✅ Resposta válida obtida na tentativa {attempt + 1}")
                        return result
                    else:
                        print(f"⚠️ Resposta inválida na tentativa {attempt + 1}: {result}")
                        
                else:
                    print(f"⚠️ Resposta vazia na tentativa {attempt + 1}")
                    
            except Exception as e:
                print(f"❌ Erro na tentativa {attempt + 1}: {e}")
                
        print(f"❌ Falha após {max_retries} tentativas")
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
    Você é um crítico especializado em análise de letras de músicas, focado em identificar e classificar elementos tóxicos em relacionamentos amorosos descritos nas letras.
    Sua tarefa é analisar a letra fornecida e responder EXATAMENTE no formato especificado abaixo, sem adicionar, omitir ou alterar campos, e seguindo rigorosamente as regras.
    
    NÍVEIS DE TOXICIDADE (escolha UM):
    muito alto - Múltiplos elementos tóxicos graves
    alto - Elementos tóxicos significativos
    moderado - Alguns elementos tóxicos problemáticos
    baixo - Poucos elementos tóxicos questionáveis
    muito baixo - Elementos tóxicos mínimos ou ambíguos
    na - Sem elementos tóxicos

    ELEMENTOS TÓXICOS A IDENTIFICAR:
    abuso_emocional: gaslighting, chantagem, manipulação, culpabilização
    ciume_possessividade: controle, posse, restrição da liberdade
    dependencia: necessidade excessiva, incapacidade de ficar sozinho
    objetificacao: redução a objeto sexual/de posse
    violencia_traicao: violência física/psicológica, ameaças

    REGRAS OBRIGATÓRIAS:
    - Primeiro, verifique se a música trata de relacionamento amoroso ou tóxico. Se NÃO tratar, o Nível de Toxicidade DEVE ser "na".
    - Se TODOS os campos abaixo forem "false", o Nível de Toxicidade DEVE ser "na".
    - Use apenas "true" ou "false" para os campos booleanos.
    - Na justificativa, cite trechos específicos da letra se houver elementos tóxicos.
    - Se não houver elementos tóxicos, escreva exatamente: "nenhum elemento tóxico identificado".
    - Não penalize amor normal, saudade ou desejo saudável.
    - Para BDSM, só considere tóxico se claramente não consensual.
    - Responda apenas no formato abaixo, sem comentários extras.

    FORMATO OBRIGATÓRIO DA RESPOSTA:
    Nivel de toxicidade: [muito alto|alto|moderado|baixo|muito baixo|na]
    abuso_emocional: [true|false]
    ciume_possessividade: [true|false]
    dependencia: [true|false]
    objetificacao: [true|false]
    violencia_traicao: [true|false]
    justificativa: [cite trechos específicos e explique OU escreva "nenhum elemento tóxico identificado"]

    Exemplos de resposta para cada nível de toxicidade:
    Sem elementos tóxicos:
    Nivel de toxicidade: na
    abuso_emocional: false
    ciume_possessividade: false
    dependencia: false
    objetificacao: false
    violencia_traicao: false
    justificativa: nenhum elemento tóxico identificado

    Muito baixo:
    Nivel de toxicidade: muito baixo
    abuso_emocional: false
    ciume_possessividade: false
    dependencia: true
    objetificacao: false
    violencia_traicao: false
    justificativa: "Não posso viver sem você" — expressa dependência leve, mas pode ser apenas uma hipérbole romântica.

    Baixo:
    Nivel de toxicidade: baixo
    abuso_emocional: false
    ciume_possessividade: true
    dependencia: false
    objetificacao: false
    violencia_traicao: false
    justificativa: "Me avisa quando sair, quero saber onde vai" — demonstra um leve controle, mas sem agressividade.

    Moderado:
    Nivel de toxicidade: moderado
    abuso_emocional: true
    ciume_possessividade: false
    dependencia: true
    objetificacao: false
    violencia_traicao: false
    justificativa: "Você me faz sentir culpado por tudo" (abuso emocional); "Eu fui longe demais por você" (dependência emocional).


    Alto:
    Nivel de toxicidade: alto
    abuso_emocional: true
    ciume_possessividade: true
    dependencia: false
    objetificacao: false
    violencia_traicao: false
    justificativa: "Por sua causa não uso mais batom, rasguei meu short curto" (controle e manipulação); "Você tem que ser só minha" (posse).

    Muito alto:
    Nivel de toxicidade: muito alto
    abuso_emocional: true
    ciume_possessividade: true
    dependencia: true
    objetificacao: true
    violencia_traicao: true
    justificativa: "Ele me bate, mas a sensação é de um beijo" (violência física romantizada); "Você só existe para me satisfazer" (objetificação); "Se me deixar, não respondo por mim" (ameaça); "Você nunca vai escapar de mim" (controle extremo); "Sem você eu não sou nada" (dependência extrema).
    
    Exemplo de música sem relação com relacionamento amoroso:
    Nivel de toxicidade: na
    abuso_emocional: false
    ciume_possessividade: false
    dependencia: false
    objetificacao: false
    violencia_traicao: false
    justificativa: nenhum elemento tóxico identificado
    
    LEMBRE-SE A CLASSIFICAÇÃO DEVE SER FEITA COM BASE NO RELACIONAMENTO AMOROSO, SE A MUSICA NAO TRATAR DE RELACIONAMENTO AMOROSO, O NIVEL DE TOXICIDADE DEVE SER "NA".
    """

    analyzer = OllamaAnalyzer()
    
    # Carrega exemplos de classificação manual
    carregar_exemplos_manuais(analyzer, '../data/30-musicas-Mozart.csv')

    resultados = []
    os.makedirs("results", exist_ok=True)

    # TESTE: Limita a 3 músicas para validar melhorias
    total = len(df)
    print(f"🧪 MODO TESTE: Analisando apenas {total} músicas para validar melhorias")
    
    for idx, row in df.head(total).iterrows():
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
        df_manual = pd.read_csv("../data/30-musicas-Mozart.csv")
        
        print("\n📊 Carregando dataset de músicas para análise...")
        df = pd.read_csv("../data/30musicas.csv")
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
