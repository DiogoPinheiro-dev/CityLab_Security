from motor.motor_asyncio import AsyncIOMotorClient

from App.settings import MONGO_DB_NAME, MONGO_DETAILS, MONGO_SERVER_SELECTION_TIMEOUT_MS

client = AsyncIOMotorClient(
    MONGO_DETAILS,
    serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
)

database = client[MONGO_DB_NAME]

colecao_alunos = database.get_collection("alunos")
colecao_logs = database.get_collection("logs")
# Referencias de rosto aprendidas no stream (FACE_LEARN_FROM_STREAM).
colecao_aprendidos = database.get_collection("rostos_aprendidos")

INDICE_RETENCAO_LOGS = "retencao_data_hora_raw"
INDICE_VALIDADE_APRENDIDOS = "validade_aprendido_em"


async def validar_conexao_mongo() -> None:
    await client.admin.command("ping")


async def _configurar_ttl(colecao, campo: str, indice: str, dias: int) -> bool:
    """Indice TTL em `campo`: o MongoDB apaga os documentos com mais de `dias`.

    Cria o indice, recria com outro prazo ou, com zero, remove o indice e para
    de apagar; devolve se o prazo ficou ligado. O MongoDB confere os prazos a
    cada minuto; ao ligar, os documentos ja vencidos saem de uma vez.
    """
    atual = (await colecao.index_information()).get(indice)
    if dias <= 0:
        if atual is not None:
            await colecao.drop_index(indice)
        return False

    segundos = dias * 24 * 60 * 60
    if atual is not None and atual.get("expireAfterSeconds") != segundos:
        # Trocar o prazo no lugar (collMod) pede permissao de administrador;
        # apagar e criar de novo so pede leitura e escrita.
        await colecao.drop_index(indice)
        atual = None
    if atual is None:
        await colecao.create_index(campo, name=indice, expireAfterSeconds=segundos)
    return True


async def configurar_retencao_logs(dias: int, colecao=colecao_logs) -> str:
    """Prazo dos eventos, pela data em data_hora_raw."""
    if await _configurar_ttl(colecao, "data_hora_raw", INDICE_RETENCAO_LOGS, dias):
        return f"eventos com mais de {dias} dias sao apagados"
    return "desligada, os eventos ficam no banco"


async def configurar_validade_aprendidos(dias: int, colecao=colecao_aprendidos) -> str:
    """Prazo das referencias aprendidas no stream, pela data em aprendido_em."""
    if await _configurar_ttl(colecao, "aprendido_em", INDICE_VALIDADE_APRENDIDOS, dias):
        return f"referencias com mais de {dias} dias sao apagadas"
    return "sem prazo, as referencias ficam no banco"
