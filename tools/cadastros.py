"""Mostra e apaga cadastros de rosto.

Cada envio da pagina de cadastro vira um documento na colecao alunos.
Recadastrar um nome cria outro documento, e o antigo continua valendo. Rode na
raiz do projeto, com o .env da API:

    python tools/cadastros.py                            # cadastros por pessoa
    python tools/cadastros.py --manter-ultimo "Fulano"   # depois de recadastrar
    python tools/cadastros.py --apagar "Fulano"          # tira a pessoa

Apagar uma pessoa leva junto as referencias aprendidas dela no stream; os
eventos ficam e saem pelo prazo de EVENT_RETENTION_DAYS. A API guarda os
cadastros na memoria: depois de apagar, reinicie a API.
"""
import argparse
import asyncio
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# O que identifica cada envio; o embedding nao sai do banco.
CAMPOS = {"nome": 1, "fotos": 1, "cadastrado_em": 1}


def _plural(quantidade: int, singular: str, plural: str) -> str:
    return f"{quantidade} {singular if quantidade == 1 else plural}"


def descrever(cadastro: dict) -> str:
    # Cadastro de antes das varias fotos nao tem o campo: era de uma foto.
    fotos = _plural(cadastro.get("fotos", 1), "foto", "fotos")
    return f"{fotos}, {cadastro.get('cadastrado_em') or 'sem data'}"


async def cadastros(alunos, filtro=None) -> dict[str, list[dict]]:
    """Cadastros por nome, do mais antigo ao mais recente."""
    por_nome = defaultdict(list)
    async for documento in alunos.find(filtro or {}, CAMPOS):
        por_nome[documento["nome"]].append(documento)
    # O _id do MongoDB cresce com a hora em que o documento entrou.
    return {nome: sorted(lista, key=lambda documento: documento["_id"])
            for nome, lista in por_nome.items()}


async def apagar_pessoa(alunos, aprendidos, nome: str) -> tuple[int, int]:
    """Apaga os cadastros e as referencias aprendidas de um nome."""
    apagados = await alunos.delete_many({"nome": nome})
    referencias = await aprendidos.delete_many({"nome": nome})
    return apagados.deleted_count, referencias.deleted_count


async def manter_ultimo(alunos, nome: str) -> tuple[int, dict | None]:
    """Apaga os cadastros antigos de um nome; devolve quantos e o que ficou."""
    lista = (await cadastros(alunos, {"nome": nome})).get(nome, [])
    if len(lista) < 2:
        return 0, (lista[-1] if lista else None)
    antigos = [documento["_id"] for documento in lista[:-1]]
    resultado = await alunos.delete_many({"_id": {"$in": antigos}})
    return resultado.deleted_count, lista[-1]


async def principal(argv=None, alunos=None, aprendidos=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    escolha = parser.add_mutually_exclusive_group()
    escolha.add_argument("--apagar", metavar="NOME",
                         help="apaga os cadastros e as referencias aprendidas de uma pessoa")
    escolha.add_argument("--manter-ultimo", metavar="NOME",
                         help="apaga os cadastros antigos de uma pessoa e deixa o mais recente")
    args = parser.parse_args(argv)
    if alunos is None or aprendidos is None:
        from Server.Db.database import colecao_alunos, colecao_aprendidos
        alunos = colecao_alunos if alunos is None else alunos
        aprendidos = colecao_aprendidos if aprendidos is None else aprendidos

    reiniciar = "Reinicie a API para ela esquecer tambem."
    nome = args.apagar or args.manter_ultimo
    if args.apagar:
        apagados, referencias = await apagar_pessoa(alunos, aprendidos, nome)
        if apagados or referencias:
            print(f"Apagados de {nome!r}: {_plural(apagados, 'cadastro', 'cadastros')} e "
                  f"{_plural(referencias, 'referencia aprendida', 'referencias aprendidas')}. "
                  f"{reiniciar}")
            return 0
    elif args.manter_ultimo:
        apagados, ficou = await manter_ultimo(alunos, nome)
        if apagados:
            print(f"Apagados de {nome!r}: "
                  f"{_plural(apagados, 'cadastro antigo', 'cadastros antigos')}. "
                  f"Fica o de {descrever(ficou)}. {reiniciar}")
            return 0
        if ficou is not None:
            print(f"{nome!r} tem um cadastro so, de {descrever(ficou)}: nada a apagar.")
            return 0
    if nome:
        print(f"Nenhum cadastro com o nome {nome!r}. Rode sem opcoes para ver os nomes.")
        return 1

    por_nome = await cadastros(alunos)
    if not por_nome:
        print("Nenhum cadastro.")
        return 0
    contagem = Counter([documento.get("nome")
                        async for documento in aprendidos.find({}, {"nome": 1})])
    for nome, lista in sorted(por_nome.items()):
        print(f"{nome}: {_plural(len(lista), 'cadastro', 'cadastros')}, "
              f"{_plural(contagem[nome], 'referencia aprendida', 'referencias aprendidas')}")
        for numero, cadastro in enumerate(lista, start=1):
            recente = " (o mais recente)" if len(lista) > 1 and numero == len(lista) else ""
            print(f"  {descrever(cadastro)}{recente}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(principal()))
