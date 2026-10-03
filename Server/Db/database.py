from motor.motor_asyncio import AsyncIOMotorClient

from App.settings import MONGO_DB_NAME, MONGO_DETAILS, MONGO_SERVER_SELECTION_TIMEOUT_MS

client = AsyncIOMotorClient(
    MONGO_DETAILS,
    serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
)

database = client[MONGO_DB_NAME]

colecao_alunos = database.get_collection("alunos")
colecao_logs = database.get_collection("logs")

INDICE_RETENCAO_LOGS = "retencao_data_hora_raw"


async def validar_conexao_mongo() -> None:
    await client.admin.command("ping")


async def configurar_retencao_logs(dias: int, colecao=colecao_logs) -> str:
    """Indice TTL em data_hora_raw: o MongoDB apaga os eventos com mais de `dias`.

    Cria o indice, recria com outro prazo ou, com zero, remove o indice e para
    de apagar. O MongoDB confere os prazos a cada minuto; ao ligar, os eventos
    ja vencidos saem de uma vez.
    """
    atual = (await colecao.index_information()).get(INDICE_RETENCAO_LOGS)
    if dias <= 0:
        if atual is not None:
            await colecao.drop_index(INDICE_RETENCAO_LOGS)
        return "desligada, os eventos ficam no banco"

    segundos = dias * 24 * 60 * 60
    if atual is not None and atual.get("expireAfterSeconds") != segundos:
        # Trocar o prazo no lugar (collMod) pede permissao de administrador;
        # apagar e criar de novo so pede leitura e escrita.
        await colecao.drop_index(INDICE_RETENCAO_LOGS)
        atual = None
    if atual is None:
        await colecao.create_index("data_hora_raw", name=INDICE_RETENCAO_LOGS,
                                   expireAfterSeconds=segundos)
    return f"eventos com mais de {dias} dias sao apagados"
