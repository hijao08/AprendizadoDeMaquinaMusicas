# Análise de Conteúdo Inapropriado em Músicas

Este projeto realiza uma análise exploratória do dataset "Top 100 Songs & Lyrics By Year (1959-2023)" do Kaggle, com foco na detecção de conteúdo inapropriado (racismo, homofobia, discurso de ódio e preconceito).

## Estrutura do Projeto

```
.
├── data/               # Dados brutos
│   └── all_songs_data.csv
├── src/               # Código fonte
│   └── exploracao_dataset.py
├── results/           # Resultados da análise
│   └── figures/       # Gráficos gerados
├── requirements.txt   # Dependências do projeto
└── README.md         # Este arquivo
```

## Requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

## Instalação

1. Clone este repositório:

```bash
git clone [URL_DO_REPOSITORIO]
cd [NOME_DO_DIRETORIO]
```

2. Crie e ative um ambiente virtual:

```bash
# No Windows
python -m venv venv
venv\Scripts\activate

# No macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso

1. Certifique-se de que o arquivo `all_songs_data.csv` está na pasta `data/`

2. Execute o script de análise:

```bash
python src/exploracao_dataset.py
```

3. Os resultados serão salvos na pasta `results/figures/`:
   - `distribuicao_temporal.png`: Distribuição de músicas por ano
   - `tamanho_letras.png`: Distribuição do tamanho das letras
   - `evolucao_termos_sensiveis.png`: Evolução dos termos sensíveis
   - `palavras_frequentes.png`: Palavras mais frequentes nas letras

## Análises Realizadas

1. **Análise Básica do Dataset**

   - Tipos de dados
   - Valores nulos
   - Valores duplicados
   - Estatísticas descritivas

2. **Análise Temporal**

   - Distribuição de músicas por ano
   - Evolução temporal das letras

3. **Análise de Conteúdo**
   - Tamanho das letras
   - Frequência de termos sensíveis
   - Palavras mais frequentes

## Contribuição

Para contribuir com o projeto:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
