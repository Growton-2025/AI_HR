"""Pluggable byte storage for candidate resumes.

Bytes live in Azure Blob in production (RESUME_STORAGE_BACKEND=azure_blob);
the Postgres backend writes candidate_resumes.file_bytes and exists so local
dev works without a storage account. Rows record which backend holds their
bytes in candidate_resumes.storage_backend, so reads always follow the row,
not the env var.
"""

import os
from typing import Optional, Protocol
from uuid import uuid4

import psycopg2

from backend.db.connection import get_db_connection, return_db_connection


class ResumeStore(Protocol):
    backend_name: str

    def put(self, *, resume_id: int, candidate_id: int, filename: str, content_type: str, data: bytes) -> str:
        """Store bytes, return the storage_key to persist on the row."""
        ...

    def get(self, *, resume_id: int, storage_key: Optional[str]) -> bytes:
        ...

    def delete(self, *, resume_id: int, storage_key: Optional[str]) -> None:
        ...


class PostgresResumeStore:
    backend_name = "postgres"

    def put(self, *, resume_id: int, candidate_id: int, filename: str, content_type: str, data: bytes) -> str:
        conn = get_db_connection()
        if not conn:
            raise RuntimeError("Database connection failed while storing resume bytes")
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE candidate_resumes SET file_bytes = %s, storage_backend = %s WHERE id = %s",
                    (psycopg2.Binary(data), self.backend_name, resume_id),
                )
            conn.commit()
            return f"pg:{resume_id}"
        finally:
            return_db_connection(conn)

    def get(self, *, resume_id: int, storage_key: Optional[str]) -> bytes:
        conn = get_db_connection()
        if not conn:
            raise RuntimeError("Database connection failed while reading resume bytes")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT file_bytes FROM candidate_resumes WHERE id = %s", (resume_id,))
                row = cur.fetchone()
            if not row or row[0] is None:
                raise FileNotFoundError(f"No stored bytes for resume {resume_id}")
            return bytes(row[0])
        finally:
            return_db_connection(conn)

    def delete(self, *, resume_id: int, storage_key: Optional[str]) -> None:
        conn = get_db_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE candidate_resumes SET file_bytes = NULL WHERE id = %s", (resume_id,))
            conn.commit()
        finally:
            return_db_connection(conn)


class AzureBlobResumeStore:
    backend_name = "azure_blob"

    def __init__(self, connection_string: str, container: str):
        # Imported lazily so the app still boots when the postgres backend is
        # selected and azure-storage-blob is not installed.
        from azure.storage.blob import BlobServiceClient

        self._service = BlobServiceClient.from_connection_string(connection_string)
        self._container = container
        self._ensure_container()

    def _ensure_container(self) -> None:
        try:
            self._service.create_container(self._container)
        except Exception:
            pass  # already exists (or race) — blob calls will surface real failures

    def _blob(self, storage_key: str):
        return self._service.get_blob_client(container=self._container, blob=storage_key)

    def put(self, *, resume_id: int, candidate_id: int, filename: str, content_type: str, data: bytes) -> str:
        from azure.storage.blob import ContentSettings

        ext = os.path.splitext(filename)[1].lower()[:16]
        storage_key = f"{candidate_id}/{resume_id}-{uuid4().hex}{ext}"
        self._blob(storage_key).upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        return storage_key

    def get(self, *, resume_id: int, storage_key: Optional[str]) -> bytes:
        if not storage_key:
            raise FileNotFoundError(f"Resume {resume_id} has no storage key")
        return self._blob(storage_key).download_blob().readall()

    def delete(self, *, resume_id: int, storage_key: Optional[str]) -> None:
        if not storage_key:
            return
        try:
            self._blob(storage_key).delete_blob()
        except Exception:
            pass  # deleting a missing blob is not an error worth surfacing


_store_cache: dict = {}


def _build_store(backend: str) -> ResumeStore:
    if backend == "azure_blob":
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
        if connection_string:
            container = os.getenv("AZURE_RESUME_CONTAINER", "resumes").strip() or "resumes"
            return AzureBlobResumeStore(connection_string, container)
        print("WARNING: RESUME_STORAGE_BACKEND=azure_blob but AZURE_STORAGE_CONNECTION_STRING is unset; using postgres bytes.")
        return PostgresResumeStore()
    return PostgresResumeStore()


def get_resume_store(backend: Optional[str] = None) -> ResumeStore:
    """Store for writes (env-selected) or for reads (pass the row's storage_backend)."""
    name = (backend or os.getenv("RESUME_STORAGE_BACKEND", "azure_blob")).strip().lower()
    if name not in _store_cache:
        _store_cache[name] = _build_store(name)
    return _store_cache[name]
