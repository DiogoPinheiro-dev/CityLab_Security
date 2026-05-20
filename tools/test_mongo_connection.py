import asyncio
import os
import socket
from pathlib import Path
from urllib.parse import urlsplit

from dns import resolver
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _redact_uri(uri: str) -> str:
    if "://" not in uri or "@" not in uri:
        return uri

    scheme, rest = uri.split("://", 1)
    _, host_part = rest.split("@", 1)
    return f"{scheme}://<usuario>:<senha>@{host_part}"


def _extract_srv_host(uri: str) -> str | None:
    if not uri.startswith("mongodb+srv://"):
        return None

    parsed = urlsplit(uri)
    return parsed.hostname


def _print_srv_records(host: str) -> None:
    query_name = f"_mongodb._tcp.{host}"
    print(f"Consultando SRV: {query_name}")
    try:
        answers = resolver.resolve(query_name, "SRV")
    except Exception as exc:
        print(f"Falha no DNS SRV: {exc}")
        return

    for answer in answers:
        print(f"SRV -> {answer.target}:{answer.port}")


async def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    uri = os.getenv("MONGO_DETAILS", "")
    db_name = os.getenv("MONGO_DB_NAME", "recon-db")
    timeout_ms = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "10000"))
    force_ipv4 = _env_bool("MONGO_FORCE_IPV4")

    print("Arquivo .env:", PROJECT_ROOT / ".env")
    print("URI carregada?", bool(uri))
    print("URI:", _redact_uri(uri) if uri else "<vazia>")
    print("DB:", db_name)
    print("Timeout ms:", timeout_ms)
    print("Forcar IPv4?", force_ipv4)

    srv_host = _extract_srv_host(uri)
    if srv_host:
        _print_srv_records(srv_host)

    options = {"serverSelectionTimeoutMS": timeout_ms}
    if force_ipv4:
        options["family"] = socket.AF_INET

    print("Conectando...")
    client = AsyncIOMotorClient(uri, **options)
    await client.admin.command("ping")
    print("Mongo OK")


if __name__ == "__main__":
    asyncio.run(main())
