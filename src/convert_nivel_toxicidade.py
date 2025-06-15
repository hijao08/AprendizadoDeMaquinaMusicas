import argparse
from pathlib import Path

import pandas as pd

# Mapeamento de categorias para valores numéricos
TOXICITY_MAPPING = {
    "na": 0.0,
    "muito baixo": 0.1,
    "baixo": 0.2,
    "moderado": 0.3,
    "alto": 0.8,
    "muito alto": 1.0,
}


def convert_toxicity(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    new_column: bool = False,
    column_name: str = "nivel_toxicidade",
) -> Path:
    """Converte a coluna ``nivel_toxicidade`` em valores numéricos.

    Parâmetros
    ----------
    input_csv: str | Path
        Caminho para o arquivo CSV de entrada.
    output_csv: str | Path | None, opcional
        Caminho para salvar o CSV convertido.  Se ``None`` (default),
        será criado um arquivo ao lado do original com sufixo ``_converted``.
    new_column: bool, opcional
        Se ``True``, mantém a coluna original e cria uma nova coluna
        ``<column_name>_num`` com os valores numéricos.  Caso contrário
        (default), a coluna original é substituída.
    column_name: str, opcional
        Nome da coluna a ser convertida.

    Retorna
    -------
    Path
        Caminho do arquivo CSV gerado.
    """

    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV de entrada não encontrado: {input_csv}")

    df = pd.read_csv(input_csv)

    if column_name not in df.columns:
        raise KeyError(
            f"A coluna '{column_name}' não foi encontrada no CSV. Colunas disponíveis: {list(df.columns)}"
        )

    # Aplica o mapeamento
    numeric_series = df[column_name].str.lower().map(TOXICITY_MAPPING)

    # Verifica se houve valores não mapeados
    if numeric_series.isna().any():
        valores_invalidos = (
            df.loc[numeric_series.isna(), column_name]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(
            "Foram encontrados valores de toxicidade sem mapeamento: "
            + ", ".join(map(str, valores_invalidos))
        )

    if new_column:
        df[f"{column_name}_num"] = numeric_series.astype(float)
    else:
        df[column_name] = numeric_series.astype(float)

    # Define o caminho de saída, se não foi fornecido
    if output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_converted{input_csv.suffix}")
    else:
        output_csv = Path(output_csv)

    # Salva o resultado
    df.to_csv(output_csv, index=False)
    return output_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Converte a coluna 'nivel_toxicidade' de categorias textuais para valores numéricos."
        )
    )
    parser.add_argument(
        "input_csv", help="Caminho para o arquivo CSV de entrada com a coluna 'nivel_toxicidade'."
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        dest="output_csv",
        default=None,
        help="Caminho para salvar o CSV convertido. Se omitido, será criado um arquivo com sufixo _converted.",
    )
    parser.add_argument(
        "--new-column",
        action="store_true",
        help="Mantém a coluna original e cria uma nova coluna numérica em vez de substituir.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_path = convert_toxicity(
        args.input_csv,
        args.output_csv,
        new_column=args.new_column,
    )

    print(f"Arquivo convertido salvo em: {output_path}")


if __name__ == "__main__":
    main() 