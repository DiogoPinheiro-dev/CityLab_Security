from motor.motor_asyncio import AsyncIOMotorClient

from App.settings import MONGO_DB_NAME, MONGO_DETAILS, MONGO_SERVER_SELECTION_TIMEOUT_MS

client = AsyncIOMotorClient(
    MONGO_DETAILS,
    serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
)

database = client[MONGO_DB_NAME]

colecao_alunos = database.get_collection("alunos")
colecao_logs = database.get_collection("logs")

async def validar_conexao_mongo() -> None:
    await client.admin.command("ping")
