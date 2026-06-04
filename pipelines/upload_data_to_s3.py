"""Seed the AWS S3 data lake with the bundled raw ESG snapshots.

Run this once to push the local ``data/raw`` JSON collections into the S3 bucket
that the ingestion layer will later read from.

    # 1. configure the target bucket + credentials
    export ESG_S3_BUCKET=my-esg-data-lake
    export AWS_REGION=us-east-1
    # (credentials via env vars, ~/.aws/credentials, or an IAM role)

    # 2. upload
    python -m pipelines.upload_data_to_s3

Afterwards, set ``ESG_DATA_BACKEND=s3`` and the pipeline / API will read straight
from S3.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crawlers.aws_s3 import S3Settings, upload_dir  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    settings = S3Settings()
    if not settings.bucket:
        raise SystemExit("ESG_S3_BUCKET is not set — point it at your target bucket first.")

    print(f"Uploading {DATA_DIR} -> s3://{settings.bucket}/{settings.prefix}/ ...")
    keys = upload_dir(DATA_DIR)
    for key in keys:
        print(f"  ✓ s3://{settings.bucket}/{key}")
    print(f"Done. {len(keys)} object(s) uploaded.")


if __name__ == "__main__":
    main()
