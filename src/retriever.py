from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

INDEX_VERSION = "v2"
EMBEDDING_MODEL_NAME = os.getenv(
    "JOBPILOT_EMBEDDING_MODEL",
    "BAAI/bge-small-zh-v1.5",
)
EMBEDDING_MODEL_REVISION = os.getenv(
    "JOBPILOT_EMBEDDING_REVISION",
    "7999e1d3359715c523056ef9478215996d62a620",
)
_MODEL_KEY = hashlib.sha1(
    f"{EMBEDDING_MODEL_NAME}@{EMBEDDING_MODEL_REVISION}".encode("utf-8")
).hexdigest()[:8]
COLLECTION_NAME = f"job_postings_{INDEX_VERSION}_{_MODEL_KEY}"
STORE_FILE = "job_documents.json"
BACKEND_FILE = "retriever_backend.json"
RRF_K = 60
VECTOR_WEIGHT = 1.0
KEYWORD_WEIGHT = 1.25


class RetrieverError(RuntimeError):
    """Raised when retrieval store operations fail."""


def _to_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def _metadata_value(value: Any) -> str:
    return "" if value is None else str(value)


def _stable_job_id(job: dict, index: int) -> str:
    return str(job.get("job_id") or f"job_{index + 1:04d}")


def build_job_documents(jobs: list[dict]) -> list[dict]:
    """Convert JobPosting dictionaries into retrievable documents."""
    documents = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            continue

        job_id = _stable_job_id(job, index)
        title = str(job.get("title") or "")
        company = str(job.get("company") or "")
        responsibilities = _to_text_list(job.get("responsibilities"))
        required_skills = _to_text_list(job.get("required_skills"))
        preferred_skills = _to_text_list(job.get("preferred_skills"))

        text_parts = [
            f"Title: {title}",
            f"Company: {company}",
            "Responsibilities: " + "; ".join(responsibilities),
            "Required skills: " + ", ".join(required_skills),
            "Preferred skills: " + ", ".join(preferred_skills),
        ]

        documents.append(
            {
                "id": job_id,
                "text": "\n".join(part for part in text_parts if part.strip()),
                "metadata": {
                    "job_id": job_id,
                    "title": title,
                    "company": company,
                    "location": _metadata_value(job.get("location")),
                },
            }
        )

    return documents


def build_retrieval_query(candidate_profile: dict, target_role: str | None = None) -> str:
    """Build a compact retrieval query from profile skills, projects, and target role."""
    if not isinstance(candidate_profile, dict):
        return target_role or ""

    parts: list[str] = []
    if target_role:
        parts.append(target_role)

    parts.extend(_to_text_list(candidate_profile.get("target_roles")))
    parts.extend(_to_text_list(candidate_profile.get("skills")))
    parts.extend(_to_text_list(candidate_profile.get("internships")))

    for project in _to_text_list_from_projects(candidate_profile.get("projects")):
        parts.append(project)

    preferences = candidate_profile.get("preferences")
    if isinstance(preferences, dict):
        parts.extend(str(value) for value in preferences.values() if value)

    return " ".join(part for part in parts if part).strip()


def _to_text_list_from_projects(projects: Any) -> list[str]:
    if not isinstance(projects, list):
        return []

    parts = []
    for project in projects:
        if not isinstance(project, dict):
            continue
        parts.extend(_to_text_list(project.get("name")))
        parts.extend(_to_text_list(project.get("description")))
        parts.extend(_to_text_list(project.get("tech_stack")))
        parts.extend(_to_text_list(project.get("highlights")))
    return parts


def _store_path(persist_dir: str | Path) -> Path:
    return Path(persist_dir) / STORE_FILE


def _backend_path(persist_dir: str | Path) -> Path:
    return Path(persist_dir) / BACKEND_FILE


def _document_hash(document: dict) -> str:
    return hashlib.sha256(str(document.get("text") or "").encode("utf-8")).hexdigest()


def _document_set_hash(documents: list[dict]) -> str:
    snapshot = sorted(
        (
            str(document.get("id") or ""),
            _document_hash(document),
        )
        for document in documents
        if isinstance(document, dict)
    )
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_manifest(backend: str, documents: list[dict], **extra: Any) -> dict[str, Any]:
    return {
        "backend": backend,
        "collection": COLLECTION_NAME if backend == "chroma" else "",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_revision": EMBEDDING_MODEL_REVISION,
        "index_version": INDEX_VERSION,
        "document_count": len(documents),
        "document_set_hash": _document_set_hash(documents),
        **extra,
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_simple_store(
    jobs: list[dict],
    documents: list[dict],
    persist_dir: str | Path,
    backend: str,
) -> dict[str, Any]:
    path = Path(persist_dir)
    jobs_by_id = {}
    for index, job in enumerate(jobs):
        if isinstance(job, dict):
            jobs_by_id[_stable_job_id(job, index)] = job

    _write_json(
        _store_path(path),
        {
            "documents": documents,
            "jobs_by_id": jobs_by_id,
        },
    )
    manifest = _index_manifest(backend, documents)
    _write_json(_backend_path(path), manifest)
    return manifest


def _load_simple_store(persist_dir: str | Path) -> dict:
    path = _store_path(persist_dir)
    if not path.exists():
        raise RetrieverError(f"检索索引文件不存在：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RetrieverError(f"检索索引 JSON 格式无效：{path}。{exc}") from exc


def _load_backend_data(persist_dir: str | Path) -> dict[str, Any]:
    path = _backend_path(persist_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_backend(persist_dir: str | Path) -> str | None:
    manifest = _load_backend_data(persist_dir)
    backend = manifest.get("backend")
    if backend == "chroma" and (
        manifest.get("embedding_model") != EMBEDDING_MODEL_NAME
        or manifest.get("embedding_revision") != EMBEDDING_MODEL_REVISION
        or manifest.get("index_version") != INDEX_VERSION
    ):
        return "simple"
    return backend if isinstance(backend, str) else None


def is_retrieval_store_current(jobs: list[dict], persist_dir: str = "data/vector_store") -> bool:
    """Return whether the persisted index matches the configured model and active jobs."""
    if not _store_path(persist_dir).exists() or not _backend_path(persist_dir).exists():
        return False

    manifest = _load_backend_data(persist_dir)
    expected_documents = build_job_documents(jobs)
    expected_hash = _document_set_hash(expected_documents)
    if (
        manifest.get("embedding_model") != EMBEDDING_MODEL_NAME
        or manifest.get("embedding_revision") != EMBEDDING_MODEL_REVISION
        or manifest.get("index_version") != INDEX_VERSION
        or manifest.get("document_set_hash") != expected_hash
        or manifest.get("document_count") != len(expected_documents)
    ):
        return False

    try:
        stored_documents = _load_simple_store(persist_dir).get("documents", [])
    except RetrieverError:
        return False
    return (
        isinstance(stored_documents, list)
        and _document_set_hash(stored_documents) == expected_hash
    )


def _import_chromadb():
    import chromadb

    return chromadb


@lru_cache(maxsize=1)
def _embedding_function():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device=os.getenv("JOBPILOT_EMBEDDING_DEVICE", "cpu"),
        normalize_embeddings=True,
        revision=EMBEDDING_MODEL_REVISION,
    )


def _chroma_metadata(document: dict) -> dict[str, str]:
    metadata = dict(document.get("metadata") or {})
    metadata["content_hash"] = _document_hash(document)
    metadata["embedding_model"] = EMBEDDING_MODEL_NAME
    metadata["embedding_revision"] = EMBEDDING_MODEL_REVISION
    metadata["index_version"] = INDEX_VERSION
    return {str(key): _metadata_value(value) for key, value in metadata.items()}


def _sync_chroma_store(documents: list[dict], persist_dir: str | Path) -> dict[str, int]:
    chromadb = _import_chromadb()
    client = chromadb.PersistentClient(path=str(Path(persist_dir)))
    embedding_function = _embedding_function()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_revision": EMBEDDING_MODEL_REVISION,
            "index_version": INDEX_VERSION,
        },
    )

    existing = collection.get(include=["metadatas"])
    existing_ids = existing.get("ids", [])
    existing_metadatas = existing.get("metadatas", [])
    existing_hashes = {
        str(document_id): str((metadata or {}).get("content_hash") or "")
        for document_id, metadata in zip(existing_ids, existing_metadatas, strict=False)
    }
    current_ids = {str(document["id"]) for document in documents}
    stale_ids = sorted(set(existing_hashes) - current_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)

    changed_documents = [
        document
        for document in documents
        if existing_hashes.get(str(document["id"])) != _document_hash(document)
    ]
    if changed_documents:
        collection.upsert(
            ids=[str(document["id"]) for document in changed_documents],
            documents=[str(document["text"]) for document in changed_documents],
            metadatas=[_chroma_metadata(document) for document in changed_documents],
        )

    return {
        "total": len(documents),
        "upserted": len(changed_documents),
        "deleted": len(stale_ids),
        "unchanged": max(0, len(documents) - len(changed_documents)),
    }


def build_chroma_store(jobs: list[dict], persist_dir: str = "data/vector_store") -> None:
    """Synchronize the local job retrieval store.

    Chroma uses a fixed multilingual embedding model and only embeds new or
    changed jobs. If Chroma or the model is unavailable, keyword retrieval
    continues to work from the lightweight JSON store.
    """
    documents = build_job_documents(jobs)
    simple_manifest = _write_simple_store(jobs, documents, persist_dir, backend="simple")
    if os.getenv("JOBPILOT_EMBEDDING_BACKEND", "chroma").strip().lower() == "simple":
        build_chroma_store.last_stats = {
            **simple_manifest,
            "total": len(documents),
            "upserted": 0,
            "deleted": 0,
            "unchanged": len(documents),
            "warning": "",
        }
        return

    try:
        stats = _sync_chroma_store(documents, persist_dir)
    except Exception as exc:
        fallback_manifest = _index_manifest("simple", documents, warning=str(exc))
        _write_json(_backend_path(persist_dir), fallback_manifest)
        build_chroma_store.last_stats = {
            **fallback_manifest,
            "total": len(documents),
            "upserted": 0,
            "deleted": 0,
            "unchanged": 0,
        }
        return

    backend_data = _index_manifest("chroma", documents, **stats)
    _write_json(_backend_path(persist_dir), backend_data)
    build_chroma_store.last_stats = backend_data


build_chroma_store.last_stats = {
    "backend": "",
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_revision": EMBEDDING_MODEL_REVISION,
    "index_version": INDEX_VERSION,
    "total": 0,
    "upserted": 0,
    "deleted": 0,
    "unchanged": 0,
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _copy_job_with_retrieval(job: dict, retrieval: dict) -> dict:
    result = dict(job)
    existing = result.get("_retrieval")
    if isinstance(existing, dict):
        merged = dict(existing)
        merged.update(retrieval)
        result["_retrieval"] = merged
    else:
        result["_retrieval"] = retrieval
    return result


def _simple_retrieve(query: str, top_k: int, persist_dir: str | Path) -> list[dict]:
    store = _load_simple_store(persist_dir)
    documents = store.get("documents", [])
    jobs_by_id = store.get("jobs_by_id", {})
    query_tokens = _tokenize(query)

    scored = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        document_id = document.get("id")
        text = str(document.get("text") or "")
        document_tokens = _tokenize(text)
        overlap = len(query_tokens & document_tokens)
        score = overlap / max(len(query_tokens), 1)
        scored.append((score, overlap, document_id, document))

    scored.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)

    results = []
    for score, _, document_id, document in scored[:top_k]:
        job = jobs_by_id.get(document_id)
        if isinstance(job, dict):
            result = dict(job)
        else:
            result = dict(document.get("metadata", {}))
            result["raw_text"] = document.get("text", "")
        result["_retrieval"] = {"backend": "simple", "score": score}
        results.append(result)

    return results


def _chroma_retrieve(query: str, top_k: int, persist_dir: str | Path) -> list[dict]:
    chromadb = _import_chromadb()
    client = chromadb.PersistentClient(path=str(Path(persist_dir)))
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=_embedding_function())
    result = collection.query(query_texts=[query], n_results=top_k, include=["metadatas", "documents", "distances"])

    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    store = _load_simple_store(persist_dir)
    jobs_by_id = store.get("jobs_by_id", {})

    retrieved = []
    for index, document_id in enumerate(ids):
        job = jobs_by_id.get(document_id)
        if isinstance(job, dict):
            item = dict(job)
        else:
            item = dict(metadatas[index] or {})
            item["raw_text"] = documents[index] if index < len(documents) else ""

        distance = distances[index] if index < len(distances) else None
        item["_retrieval"] = {"backend": "chroma", "distance": distance}
        retrieved.append(item)

    return retrieved


def retrieve_jobs(query: str, top_k: int = 10, persist_dir: str = "data/vector_store") -> list[dict]:
    """Retrieve top-k relevant jobs from the local retrieval store."""
    if top_k <= 0:
        return []

    backend = _load_backend(persist_dir)
    if backend == "chroma":
        try:
            return _chroma_retrieve(query, top_k, persist_dir)
        except Exception:
            return _simple_retrieve(query, top_k, persist_dir)

    return _simple_retrieve(query, top_k, persist_dir)


def _tfidf_scores(query: str, texts: list[str]) -> list[float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        query_tokens = _tokenize(query)
        scores = []
        for text in texts:
            document_tokens = _tokenize(text)
            scores.append(len(query_tokens & document_tokens) / max(len(query_tokens), 1))
        return scores

    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w#+.-]+\b", lowercase=True)
        document_matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([query])
        return (document_matrix @ query_vector.T).toarray().ravel().tolist()
    except Exception:
        query_tokens = _tokenize(query)
        scores = []
        for text in texts:
            document_tokens = _tokenize(text)
            scores.append(len(query_tokens & document_tokens) / max(len(query_tokens), 1))
        return scores


def simple_bm25_retrieve(query: str, jobs: list[dict], top_k: int = 10) -> list[dict]:
    """BM25-like keyword retrieval using TF-IDF for exact keyword recall."""
    if top_k <= 0 or not query.strip():
        return []

    valid_jobs = [job for job in jobs if isinstance(job, dict)]
    documents = build_job_documents(valid_jobs)
    if not documents:
        return []

    scores = _tfidf_scores(query, [document["text"] for document in documents])
    scored = [
        (float(score), index)
        for index, score in enumerate(scores)
        if score and score > 0
    ]
    scored.sort(key=lambda item: (item[0], documents[item[1]]["id"]), reverse=True)

    results = []
    for score, index in scored[:top_k]:
        job = valid_jobs[index]
        results.append(
            _copy_job_with_retrieval(
                job,
                {
                    "backend": "keyword",
                    "keyword_score": score,
                },
            )
        )

    return results


def _merge_hybrid_results(
    vector_results: list[dict],
    keyword_results: list[dict],
    top_k: int,
    rrf_k: int = RRF_K,
    vector_weight: float = VECTOR_WEIGHT,
    keyword_weight: float = KEYWORD_WEIGHT,
) -> tuple[list[dict], int]:
    if top_k <= 0:
        return [], 0
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if vector_weight < 0 or keyword_weight < 0:
        raise ValueError("RRF weights must be non-negative")

    merged: dict[str, dict] = {}
    order: dict[str, dict[str, int]] = {}
    sources: dict[str, set[str]] = {}

    def add_result(job: dict, source: str, rank: int) -> None:
        if not isinstance(job, dict):
            return
        job_id = job.get("job_id")
        if not job_id:
            return

        if job_id not in merged:
            merged[job_id] = dict(job)
            retrieval = job.get("_retrieval")
            if isinstance(retrieval, dict):
                merged[job_id]["_retrieval"] = dict(retrieval)
            order[job_id] = {}
            sources[job_id] = set()
        else:
            existing_retrieval = merged[job_id].get("_retrieval")
            new_retrieval = job.get("_retrieval")
            if isinstance(existing_retrieval, dict) and isinstance(new_retrieval, dict):
                existing_retrieval.update(new_retrieval)
            elif isinstance(new_retrieval, dict):
                merged[job_id]["_retrieval"] = dict(new_retrieval)

        rank_position = rank + 1
        previous_rank = order[job_id].get(source)
        if previous_rank is None or rank_position < previous_rank:
            order[job_id][source] = rank_position
        sources[job_id].add(source)

    for rank, job in enumerate(vector_results):
        add_result(job, "vector", rank)
    for rank, job in enumerate(keyword_results):
        add_result(job, "keyword", rank)

    for job_id, item in merged.items():
        source_set = sources[job_id]
        if source_set == {"vector", "keyword"}:
            item["retrieve_source"] = "both"
        elif source_set == {"keyword"}:
            item["retrieve_source"] = "keyword"
        else:
            item["retrieve_source"] = "vector"
        ranks = order[job_id]
        vector_rank = ranks.get("vector")
        keyword_rank = ranks.get("keyword")
        hybrid_score = 0.0
        if vector_rank is not None:
            hybrid_score += float(vector_weight) / (rrf_k + vector_rank)
        if keyword_rank is not None:
            hybrid_score += float(keyword_weight) / (rrf_k + keyword_rank)
        item["vector_rank"] = vector_rank
        item["keyword_rank"] = keyword_rank
        item["hybrid_score"] = round(hybrid_score, 8)
        retrieval = item.setdefault("_retrieval", {})
        if isinstance(retrieval, dict):
            retrieval.update(
                {
                    "vector_rank": vector_rank,
                    "keyword_rank": keyword_rank,
                    "hybrid_score": item["hybrid_score"],
                    "rrf_k": rrf_k,
                    "vector_weight": vector_weight,
                    "keyword_weight": keyword_weight,
                }
            )

    def sort_key(item: dict) -> tuple[float, int, int, str]:
        job_id = item.get("job_id", "")
        ranks = order.get(job_id, {})
        return (
            -float(item.get("hybrid_score", 0.0)),
            ranks.get("keyword", 10_000),
            ranks.get("vector", 10_000),
            str(job_id),
        )

    sorted_results = sorted(merged.values(), key=sort_key)
    return sorted_results[:top_k], len(merged)


def hybrid_retrieve(
    query: str,
    jobs: list[dict],
    top_k: int = 10,
    persist_dir: str = "data/vector_store",
) -> list[dict]:
    """Retrieve jobs with vector search plus keyword recall, then merge by job_id."""
    if top_k <= 0:
        hybrid_retrieve.last_stats = {
            "query": query,
            "vector_top_k": top_k,
            "keyword_top_k": top_k,
            "vector_result_count": 0,
            "keyword_result_count": 0,
            "merged_count": 0,
            "final_retrieved_count": 0,
            "vector_error": "",
            "keyword_error": "",
            "fusion": "rrf",
            "rrf_k": RRF_K,
            "vector_weight": VECTOR_WEIGHT,
            "keyword_weight": KEYWORD_WEIGHT,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "index_version": INDEX_VERSION,
        }
        return []

    vector_error = ""
    try:
        vector_results = retrieve_jobs(query=query, top_k=top_k, persist_dir=persist_dir)
    except Exception as exc:
        vector_results = []
        vector_error = str(exc)

    keyword_error = ""
    try:
        keyword_results = simple_bm25_retrieve(query=query, jobs=jobs, top_k=top_k)
    except Exception as exc:
        keyword_results = []
        keyword_error = str(exc)

    merged_results, merged_count = _merge_hybrid_results(vector_results, keyword_results, top_k=top_k)

    hybrid_retrieve.last_stats = {
        "query": query,
        "vector_top_k": top_k,
        "keyword_top_k": top_k,
        "vector_result_count": len(vector_results),
        "keyword_result_count": len(keyword_results),
        "merged_count": merged_count,
        "final_retrieved_count": len(merged_results),
        "vector_error": vector_error,
        "keyword_error": keyword_error,
        "fusion": "rrf",
        "rrf_k": RRF_K,
        "vector_weight": VECTOR_WEIGHT,
        "keyword_weight": KEYWORD_WEIGHT,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "index_version": INDEX_VERSION,
    }
    return merged_results


hybrid_retrieve.last_stats = {
    "query": "",
    "vector_top_k": 0,
    "keyword_top_k": 0,
    "vector_result_count": 0,
    "keyword_result_count": 0,
    "merged_count": 0,
    "final_retrieved_count": 0,
    "vector_error": "",
    "keyword_error": "",
    "fusion": "rrf",
    "rrf_k": RRF_K,
    "vector_weight": VECTOR_WEIGHT,
    "keyword_weight": KEYWORD_WEIGHT,
    "embedding_model": EMBEDDING_MODEL_NAME,
    "index_version": INDEX_VERSION,
}


class ChromaRetriever:
    """Small class wrapper kept for future retriever extension."""

    def __init__(self, persist_dir: str | Path) -> None:
        self.persist_dir = Path(persist_dir)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return retrieve_jobs(query=query, top_k=top_k, persist_dir=str(self.persist_dir))
