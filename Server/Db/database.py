import os
import socket
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "Server" / ".env", override=True)

MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "recon-db")
MONGO_SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "10000"))
MONGO_FORCE_IPV4 = os.getenv("MONGO_FORCE_IPV4", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

client_options = {
    "serverSelectionTimeoutMS": MONGO_SERVER_SELECTION_TIMEOUT_MS,
}
if MONGO_FORCE_IPV4:
    client_options["family"] = socket.AF_INET

client = AsyncIOMotorClient(
    MONGO_DETAILS,
    **client_options,
)

database = client[MONGO_DB_NAME]

colecao_alunos = database.get_collection("alunos")
colecao_logs = database.get_collection("logs")

async def validar_conexao_mongo() -> None:
    await client.admin.command("ping")
