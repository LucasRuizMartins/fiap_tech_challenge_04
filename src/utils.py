import boto3
import dotenv
import os 
import pandas as pd 
from io import BytesIO

from pathlib import Path
import sys
import subprocess
import importlib
from pathlib import Path


# Garante que o .env seja encontrado independente de onde o notebook está rodando
caminho_env = Path(__file__).resolve().parent.parent / '.env'
if not caminho_env.exists():
    caminho_env = Path(__file__).resolve().parent.parent.parent / '.env'
dotenv.load_dotenv(caminho_env, override=True)

# Criando a cessão 

def iniciar_cessao_aws():
    ID_CONTA = os.getenv("ID_CONTA")
    AWS_REGION = os.getenv("AWS_REGION")
    AWS_ACCESS_KEY_ID=os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")


    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    return session



def carregar_parquet_s3(
    s3_client,
    bucket: str,
    nome_tabela: str,
    camada: str = "gold",
    ano: str | int = None,
    subpasta: str = None,
    ler_dicionario: bool = False,
) -> pd.DataFrame:
    """
    Carrega qualquer arquivo Parquet do S3 de forma flexível.
    
    Exemplos de chaves geradas:
    - Com ano:        'gold/ano=historico/dados/TS_ALUNO.parquet'
    - Dados externos: 'gold/dados_externos/fact_escola.parquet'
    - Fato 2026:      'gold/fato/ano=2026/dados/FATO_ALFABETIZACAO.parquet'
    """
    nome_arquivo = f"dicionario_{nome_tabela}.parquet" if ler_dicionario else f"{nome_tabela}.parquet"
    
    # Montagem dinâmica da chave no S3
    if subpasta:
        # Se você passar uma subpasta direta (ex: 'dados_externos' ou 'fato/ano=2026/dados')
        chave = f"{camada}/{subpasta}/{nome_arquivo}"
    elif ano is not None:
        # Padrão com partição de ano
        pasta_interna = "dicionario" if ler_dicionario else "dados"
        chave = f"{camada}/ano={ano}/{pasta_interna}/{nome_arquivo}"
    else:
        # Padrão direto na camada
        chave = f"{camada}/{nome_arquivo}"

    print(f"Lendo: s3://{bucket}/{chave}")

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=chave)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"Arquivo não encontrado:\ns3://{bucket}/{chave}")
    except Exception as e:
        raise RuntimeError(f"Erro ao ler {chave} do bucket {bucket}.\n{e}")




dependencias = {
    "pandas": "pandas",
    "openpyxl": "openpyxl",   # necessário para ler .xlsx
    "xlrd": "xlrd>=2.0.1",    # necessário para ler .xls (Tabela 2.2 e 4.3)
    "pyarrow": "pyarrow",     # necessário para salvar/ler .parquet
}

def salvar_parquet_s3(df: pd.DataFrame, s3_client, bucket: str, chave: str):
    """
    Higieniza o DataFrame e salva em .parquet direto no bucket S3 via buffer em memória.
    """
    df = df.rename(columns=str).copy()
    for col in df.columns:
        if df[col].dtype == "object":
            nulos = df[col].isna()
            df[col] = df[col].astype(str)
            df.loc[nulos, col] = None

    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3_client.put_object(Bucket=bucket, Key=chave, Body=buffer.getvalue())
    print(f"Salvo no S3: s3://{bucket}/{chave}")


def listar_arquivos_s3(s3_client, bucket: str, prefixo: str = "") -> list[str]:
    """
    Lista as chaves de todos os arquivos disponíveis em um bucket S3 sob determinado prefixo (pasta).
    """
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefixo)
    chaves = [obj["Key"] for obj in response.get("Contents", []) if not obj["Key"].endswith("/")]
    return chaves


