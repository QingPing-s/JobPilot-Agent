from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


COLLECTION_NAME = "job_postings"
STORE_FILE = "job_documents.json"
BACKEND_FILE = "retriever_backend.json"


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


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_simple_store(jobs: list[dict], documents: list[dict], persist_dir: str | Path, backend: str) -> None:
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
    _write_json(_backend_path(path), {"backend": backend})


def _load_simple_store(persist_dir: str | Path) -> dict:
    path = _store_path(persist_dir)
    if not path.exists():
        raise RetrieverError(f"检索索引文件不存在：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RetrieverError(f"检索索引 JSON 格式无效：{path}。{exc}") from exc


def _load_backend(persist_dir: str | Path) -> str | None:
    path = _backend_path(persist_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    backend = data.get("backend")
    return backend if isinstance(backend, str) else None


def _import_chromadb():
    import chromadb

    return chromadb


def _build_chroma_store(documents: list[dict], persist_dir: str | Path) -> None:
    chromadb = _import_chromadb()
    client = chromadb.PersistentClient(path=str(Path(persist_dir)))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)
    if not documents:
        return

    collection.add(
        ids=[document["id"] for document in documents],
        documents=[document["text"] for document in documents],
        metadatas=[document["metadata"] for document in documents],
    )


def build_chroma_store(jobs: list[dict], persist_dir: str = "data/vector_store") -> None:
    """Build a local job retrieval store.

    ChromaDB is used when available. If ChromaDB or its default embedding
    backend is unavailable, a simple lexical store is written instead.
    """
    documents = build_job_documents(jobs)
    _write_simple_store(jobs, documents, persist_dir, backend="simple")

    try:
        _build_chroma_store(documents, persist_dir)
    except Exception:
        return

    _write_json(_backend_path(persist_dir), {"backend": "chroma"})


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
    collection = client.get_collection(name=COLLECTION_NAME)
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


def _merge_hybrid_results(vector_results: list[dict], keyword_results: list[dict], top_k: int) -> tuple[list[dict], int]:
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
            order[job_id] = {}
            sources[job_id] = set()
        else:
            existing_retrieval = merged[job_id].get("_retrieval")
            new_retrieval = job.get("_retrieval")
            if isinstance(existing_retrieval, dict) and isinstance(new_retrieval, dict):
                existing_retrieval.update(new_retrieval)

        order[job_id][source] = rank
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

    source_priority = {"both": 0, "keyword": 1, "vector": 2}

    def sort_key(item: dict) -> tuple[int, int, int, str]:
        job_id = item.get("job_id", "")
        ranks = order.get(job_id, {})
        return (
            source_priority.get(item.get("retrieve_source"), 99),
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
            "merged_count": 0,
            "final_retrieved_count": 0,
        }
        return []

    try:
        vector_results = retrieve_jobs(query=query, top_k=top_k, persist_dir=persist_dir)
    except Exception:
        vector_results = []

    keyword_results = simple_bm25_retrieve(query=query, jobs=jobs, top_k=top_k)
    merged_results, merged_count = _merge_hybrid_results(vector_results, keyword_results, top_k=top_k)

    hybrid_retrieve.last_stats = {
        "query": query,
        "vector_top_k": top_k,
        "keyword_top_k": top_k,
        "merged_count": merged_count,
        "final_retrieved_count": len(merged_results),
    }
    return merged_results


hybrid_retrieve.last_stats = {
    "query": "",
    "vector_top_k": 0,
    "keyword_top_k": 0,
    "merged_count": 0,
    "final_retrieved_count": 0,
}


class ChromaRetriever:
    """Small class wrapper kept for future retriever extension."""

    def __init__(self, persist_dir: str | Path) -> None:
        self.persist_dir = Path(persist_dir)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return retrieve_jobs(query=query, top_k=top_k, persist_dir=str(self.persist_dir))
