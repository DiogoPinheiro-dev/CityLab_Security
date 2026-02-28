from motor.motor_asyncio import AsyncIOMotorClient

# URL de conexão padrão do MongoDB local
MONGO_DETAILS = "mongodb://localhost:27017"

client = AsyncIOMotorClient(MONGO_DETAILS)

# Cria/Acessa o banco de dados
database = client["recon-db"]

# Coleções (equivalentes a tabelas)
colecao_alunos = database.get_collection("alunos")
colecao_logs = database.get_collection("logs")