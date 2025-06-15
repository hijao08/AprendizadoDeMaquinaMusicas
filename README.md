# Análise de Relacionamentos Tóxicos em Músicas 🎵

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Sobre 📖

Este projeto realiza uma análise exploratória aprofundada do dataset "Top 100 Songs & Lyrics By Year (1959-2023)" do Kaggle, com foco na identificação e análise de conteúdo relacionado a relacionamentos tóxicos nas letras das músicas. A análise abrange:

- Detecção de abuso emocional
- Identificação de ciúme e possessividade
- Análise de dependência emocional
- Mapeamento de objetificação
- Identificação de violência e traição

O objetivo é compreender como os relacionamentos tóxicos são retratados na música popular ao longo do tempo, analisando padrões, tendências e a evolução deste tipo de conteúdo nas letras.

## Estrutura do Projeto 📁

```
.
├── data/               # Dados brutos e processados
│   └── all_songs_data.csv
├── src/               # Código fonte do projeto
│   ├── analise_resultados.ipynb    # Notebook de análise dos resultados
│   ├── ollama_analysis.py          # Script de análise com modelo de linguagem
│   ├── convert_nivel_toxicidade.py # Processamento dos níveis de toxicidade
│   └── core/                       # Módulos principais
├── requirements.txt   # Dependências do projeto
└── README.md         # Documentação
```

## Requisitos 🛠️

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Dependências Principais

- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- nltk >= 3.8.0
- requests >= 2.31.0
- ollama >= 0.1.0

## Instalação 💻

1. Clone este repositório:

```bash
git clone https://github.com/seu-usuario/AprendizadoDeMaquinaMusica.git
cd AprendizadoDeMaquinaMusica
```

2. Crie e ative um ambiente virtual:

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso 🚀

1. Prepare os dados:

   - Certifique-se de que o arquivo `all_songs_data.csv` está na pasta `data/`

2. Execute as análises:

   - Abra o notebook `src/analise_resultados.ipynb` usando Jupyter Notebook ou JupyterLab
   - Execute as células em sequência para reproduzir as análises

3. Para análises específicas:
   - Use `ollama_analysis.py` para análises de conteúdo com modelo de linguagem
   - Execute `convert_nivel_toxicidade.py` para processamento dos níveis de toxicidade

## Análises Realizadas 📊

### 1. Análise Básica do Dataset

- Exploração da estrutura dos dados
- Tratamento de valores nulos e duplicados
- Estatísticas descritivas detalhadas

### 2. Análise Temporal

- Evolução dos relacionamentos tóxicos nas músicas ao longo das décadas
- Tendências temporais em diferentes categorias de toxicidade

### 3. Análise de Conteúdo

- Classificação dos níveis de toxicidade
- Identificação de padrões de relacionamentos abusivos
- Análise de diferentes tipos de comportamentos tóxicos

## Resultados 📈

Os principais insights obtidos incluem:

- Evolução temporal dos relacionamentos tóxicos na música
- Distribuição dos diferentes tipos de toxicidade
- Análises comparativas por década
- Tendências e padrões identificados nos relacionamentos retratados

## Contribuição 🤝

Para contribuir com o projeto:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Faça commit das mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## Licença 📝

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Contato 📧

Para dúvidas ou sugestões, sinta-se à vontade para abrir uma issue ou enviar um pull request.
