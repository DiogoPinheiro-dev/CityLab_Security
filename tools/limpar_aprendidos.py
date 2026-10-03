"""Mostra e apaga as referencias de rosto aprendidas no stream.

Com FACE_LEARN_FROM_STREAM=1, a API guarda na colecao rostos_aprendidos ate
FACE_LEARNED_PER_PERSON referencias por pessoa. Rode na raiz do projeto, com o
.env da API:

    python tools/limpar_aprendidos.py                  # quantas por pessoa
    python tools/limpar_aprendidos.py --nome "Fulano"  # apaga as de uma pessoa
    python tools/limpar_aprendidos.py --tudo           # apaga todas

A API tambem guarda as referencias na memoria: depois de apagar, reinicie a API.
"""
import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def resumo(colecao) -> Counter:
    """Quantas referencias cada pessoa tem, so pelo nome."""
    return Counter([documento.get("nome") async for documento in colecao.find({}, {"nome": 1})])


async def apagar(colecao, nome=None) -> int:
    resultado = await colecao.delete_many({"nome": nome} if nome else {})
    return resultado.deleted_count


async def principal(argv=None, colecao=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    escolha = parser.add_mutually_exclusive_group()
    escolha.add_argument("--nome", help="apaga as referencias de uma pessoa, pelo nome do cadastro")
    escolha.add_argument("--tudo", action="store_true", help="apaga todas as referencias")
    args = parser.parse_args(argv)
    if colecao is None:
        from Server.Db.database import colecao_aprendidos as colecao

    if args.nome or args.tudo:
        apagadas = await apagar(colecao, args.nome)
        print(f"{apagadas} referencias apagadas. Reinicie a API para ela esquecer tambem.")
        return 0

    contagem = await resumo(colecao)
    if not contagem:
        print("Nenhuma referencia aprendida.")
    for nome, total in sorted(contagem.items()):
        print(f"{total}  {nome}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(principal()))
