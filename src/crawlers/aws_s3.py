"""AWS S3 connectivity for the ingestion layer.

The platform's raw ESG collections (``companies`` / ``supply_chain`` /
``controversies`` / ``esg_documents``) can live in an **S3 data lake** instead of
the local ``data/raw`` folder. This mirrors the reference architecture where
Scrapy/Kafka land raw filings in cloud object storage before the pipeline
consumes them.

Activation is driven entirely by environment variables (see :class:`S3Settings`).
When ``ESG_DATA_BACKEND`` is not ``s3`` nothing here touches the network and
``boto3`` is never imported, so the project still runs with zero AWS dependencies
or credentials.

    export ESG_DATA_BACKEND=s3
    export ESG_S3_BUCKET=my-esg-data-lake
    export ESG_S3_PREFIX=data/raw          # optional, this is the default
    export AWS_REGION=us-east-1
    # credentials via the standard AWS chain (env vars, ~/.aws/credentials, IAM role)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class S3Settings:
    """Resolve S3 configuration from environment variables."""

    def __init__(self) -> None:
        self.backend = os.getenv("ESG_DATA_BACKEND", "local").lower()
        self.bucket = os.getenv("ESG_S3_BUCKET", "")
        self.prefix = os.getenv("ESG_S3_PREFIX", "data/raw").strip("/")
        self.output_prefix = os.getenv("ESG_S3_OUTPUT_PREFIX", "data/curated").strip("/")
        self.region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

    @property
    def enabled(self) -> bool:
        """True when the ingestion layer should read from S3."""
        return self.backend == "s3"

    def key_for(self, name: str) -> str:
        """S3 key for a raw input collection."""
        return f"{self.prefix}/{name}" if self.prefix else name

    def output_key_for(self, name: str) -> str:
        """S3 key for a curated output artifact."""
        return f"{self.output_prefix}/{name}" if self.output_prefix else name


def _client():
    """Lazily build a boto3 S3 client (imported only when S3 is used)."""
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "ESG_DATA_BACKEND=s3 requires boto3. Install it with `pip install boto3`."
        ) from exc
    return boto3.client("s3", region_name=S3Settings().region)


def read_json(name: str) -> Any:
    """Read and parse a single JSON object from the configured S3 bucket."""
    settings = S3Settings()
    if not settings.bucket:
        raise RuntimeError("ESG_DATA_BACKEND=s3 but ESG_S3_BUCKET is not set.")
    obj = _client().get_object(Bucket=settings.bucket, Key=settings.key_for(name))
    return json.loads(obj["Body"].read().decode("utf-8"))


def write_json(name: str, data: Any) -> str:
    """Write a curated result artifact as JSON to the S3 output prefix.

    Returns the S3 key written. Used to persist computed ESG scores, narratives
    and SFDR classifications back to the data lake's curated layer.
    """
    settings = S3Settings()
    if not settings.bucket:
        raise RuntimeError("ESG_S3_BUCKET is not set.")
    key = settings.output_key_for(name)
    body = json.dumps(data, indent=2, default=str).encode("utf-8")
    _client().put_object(
        Bucket=settings.bucket, Key=key, Body=body, ContentType="application/json"
    )
    return key


def upload_dir(local_dir: str | Path) -> list[str]:
    """Upload every ``*.json`` file in ``local_dir`` to the S3 data lake.

    Returns the list of S3 keys written — used to seed the bucket from the
    bundled ``data/raw`` snapshots.
    """
    settings = S3Settings()
    if not settings.bucket:
        raise RuntimeError("ESG_S3_BUCKET is not set.")
    client = _client()
    written: list[str] = []
    for path in sorted(Path(local_dir).glob("*.json")):
        key = settings.key_for(path.name)
        client.upload_file(str(path), settings.bucket, key)
        written.append(key)
    return written
