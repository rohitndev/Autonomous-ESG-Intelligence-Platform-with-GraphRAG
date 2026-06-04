"""Write the platform's curated results back to the AWS S3 data lake.

Runs the full pipeline once and persists the outputs (portfolio scores,
per-company ESG profiles, graph stats) to ``s3://<bucket>/data/curated/`` —
closing the AWS loop: raw data is read from S3, results are written back to S3.

    export ESG_DATA_BACKEND=s3            # so raw data is read from S3 too (optional)
    export ESG_S3_BUCKET=my-esg-data-lake
    export ESG_S3_OUTPUT_PREFIX=data/curated   # optional (this is the default)
    export AWS_REGION=us-east-1
    python -m pipelines.export_results_to_s3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crawlers.aws_s3 import S3Settings  # noqa: E402
from src.engine import ESGEngine  # noqa: E402


def main() -> None:
    settings = S3Settings()
    if not settings.bucket:
        raise SystemExit("ESG_S3_BUCKET is not set — point it at your target bucket first.")

    print("Running pipeline and exporting curated results to S3 ...")
    engine = ESGEngine()
    keys = engine.export_results_to_s3()
    for key in keys:
        print(f"  ✓ s3://{settings.bucket}/{key}")
    print(f"Done. {len(keys)} artifact(s) written to s3://{settings.bucket}/{settings.output_prefix}/.")


if __name__ == "__main__":
    main()
