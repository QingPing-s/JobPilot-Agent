from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.job_store import DEFAULT_DB_PATH, list_jobs, list_parsed_jobs, upsert_jobs
from src.retriever import build_chroma_store

SEED_PATH = ROOT / "data" / "job_seed.json"
DB_PATH = ROOT / DEFAULT_DB_PATH
VECTOR_PATH = ROOT / "data" / "vector_store"


def main() -> None:
    records = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    if not list_jobs(db_path=DB_PATH):
        upsert_jobs(
            jd_texts=[item["raw_text"] for item in records],
            jd_filenames=[item.get("filename", "seed.txt") for item in records],
            source="reproducible_seed",
            db_path=DB_PATH,
        )
    jobs = list_parsed_jobs(db_path=DB_PATH)
    build_chroma_store(jobs, persist_dir=str(VECTOR_PATH))
    print(f"Initialized {len(jobs)} jobs and refreshed {VECTOR_PATH}.")


if __name__ == "__main__":
    main()
