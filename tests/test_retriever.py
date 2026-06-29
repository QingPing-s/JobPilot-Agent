from src import retriever


def _jobs():
    return [
        {
            "job_id": "job_agent",
            "title": "AI Agent Intern",
            "company": "Agentic AI",
            "location": "Remote",
            "responsibilities": ["Build tool calling workflows"],
            "required_skills": ["Python", "LLM"],
            "preferred_skills": ["RAG", "LangGraph"],
            "raw_text": "Agent JD",
        },
        {
            "job_id": "job_backend",
            "title": "Backend Intern",
            "company": "Data Systems",
            "location": "Shanghai",
            "responsibilities": ["Build backend APIs"],
            "required_skills": ["Java", "SQL"],
            "preferred_skills": ["Docker"],
            "raw_text": "Backend JD",
        },
    ]


def test_build_job_documents():
    documents = retriever.build_job_documents(_jobs())

    assert documents[0]["id"] == "job_agent"
    assert "AI Agent Intern" in documents[0]["text"]
    assert "Python" in documents[0]["text"]
    assert documents[0]["metadata"] == {
        "job_id": "job_agent",
        "title": "AI Agent Intern",
        "company": "Agentic AI",
        "location": "Remote",
    }


def test_build_retrieval_query():
    query = retriever.build_retrieval_query(
        {
            "skills": ["Python", "RAG"],
            "target_roles": ["LLM Application Intern"],
            "projects": [
                {
                    "name": "Agent Demo",
                    "description": "Tool calling prototype",
                    "tech_stack": ["LangGraph"],
                    "highlights": ["Traced agent execution"],
                }
            ],
            "internships": ["AI platform intern"],
            "preferences": {"location": "Remote"},
        },
        target_role="AI Agent Intern",
    )

    assert "AI Agent Intern" in query
    assert "Python" in query
    assert "LangGraph" in query
    assert "Remote" in query


def test_simple_retrieval_fallback(monkeypatch, tmp_path):
    def fake_import_chromadb():
        raise ImportError("chromadb not installed")

    monkeypatch.setattr(retriever, "_import_chromadb", fake_import_chromadb)

    retriever.build_chroma_store(_jobs(), persist_dir=str(tmp_path))
    results = retriever.retrieve_jobs("AI Agent Python RAG", top_k=1, persist_dir=str(tmp_path))

    assert len(results) == 1
    assert results[0]["job_id"] == "job_agent"
    assert results[0]["_retrieval"]["backend"] == "simple"


def test_simple_bm25_retrieve_matches_exact_keywords():
    results = retriever.simple_bm25_retrieve("LangGraph DeepSeek ChromaDB", _jobs(), top_k=2)

    assert results
    assert results[0]["job_id"] == "job_agent"
    assert results[0]["_retrieval"]["backend"] == "keyword"
    assert results[0]["_retrieval"]["keyword_score"] > 0


def test_hybrid_retrieve_merges_and_marks_sources(monkeypatch):
    def fake_retrieve_jobs(query, top_k, persist_dir):
        return [
            {"job_id": "job_agent", "title": "AI Agent Intern", "_retrieval": {"backend": "chroma"}},
            {"job_id": "job_vector_only", "title": "Vector Only", "_retrieval": {"backend": "chroma"}},
        ]

    def fake_keyword_retrieve(query, jobs, top_k):
        return [
            {"job_id": "job_agent", "title": "AI Agent Intern", "_retrieval": {"backend": "keyword"}},
            {"job_id": "job_keyword_only", "title": "Keyword Only", "_retrieval": {"backend": "keyword"}},
        ]

    monkeypatch.setattr(retriever, "retrieve_jobs", fake_retrieve_jobs)
    monkeypatch.setattr(retriever, "simple_bm25_retrieve", fake_keyword_retrieve)

    results = retriever.hybrid_retrieve("LangGraph", _jobs(), top_k=10, persist_dir="unused")
    by_id = {result["job_id"]: result for result in results}

    assert set(by_id) == {"job_agent", "job_vector_only", "job_keyword_only"}
    assert by_id["job_agent"]["retrieve_source"] == "both"
    assert by_id["job_vector_only"]["retrieve_source"] == "vector"
    assert by_id["job_keyword_only"]["retrieve_source"] == "keyword"
    assert by_id["job_agent"]["vector_rank"] == 1
    assert by_id["job_agent"]["keyword_rank"] == 1
    assert by_id["job_agent"]["hybrid_score"] > by_id["job_vector_only"]["hybrid_score"]
    assert retriever.hybrid_retrieve.last_stats["merged_count"] == 3
    assert retriever.hybrid_retrieve.last_stats["final_retrieved_count"] == 3


def test_hybrid_retrieve_uses_keyword_when_vector_fails(monkeypatch):
    def fake_retrieve_jobs(query, top_k, persist_dir):
        raise RuntimeError("vector unavailable")

    monkeypatch.setattr(retriever, "retrieve_jobs", fake_retrieve_jobs)

    results = retriever.hybrid_retrieve("LangGraph Python", _jobs(), top_k=1, persist_dir="missing")

    assert len(results) == 1
    assert results[0]["job_id"] == "job_agent"
    assert results[0]["retrieve_source"] == "keyword"


def test_chroma_retriever_wrapper(monkeypatch, tmp_path):
    def fake_retrieve_jobs(query, top_k, persist_dir):
        return [{"job_id": "job_agent", "query": query, "top_k": top_k}]

    monkeypatch.setattr(retriever, "retrieve_jobs", fake_retrieve_jobs)

    wrapper = retriever.ChromaRetriever(tmp_path)
    results = wrapper.search("Python", top_k=3)

    assert results == [{"job_id": "job_agent", "query": "Python", "top_k": 3}]


def test_incremental_chroma_sync_only_upserts_changed_documents(monkeypatch, tmp_path):
    class FakeCollection:
        def __init__(self):
            self.upserted = []
            self.deleted = []

        def get(self, include):
            return {
                "ids": ["job_agent", "job_stale"],
                "metadatas": [
                    {"content_hash": retriever._document_hash(retriever.build_job_documents(_jobs())[0])},
                    {"content_hash": "old"},
                ],
            }

        def upsert(self, **kwargs):
            self.upserted.extend(kwargs["ids"])

        def delete(self, ids):
            self.deleted.extend(ids)

    collection = FakeCollection()

    class FakeClient:
        def get_or_create_collection(self, **kwargs):
            return collection

    class FakeChroma:
        @staticmethod
        def PersistentClient(path):
            return FakeClient()

    monkeypatch.setattr(retriever, "_import_chromadb", lambda: FakeChroma)
    monkeypatch.setattr(retriever, "_embedding_function", lambda: object())

    stats = retriever._sync_chroma_store(
        retriever.build_job_documents(_jobs()),
        tmp_path,
    )

    assert collection.upserted == ["job_backend"]
    assert collection.deleted == ["job_stale"]
    assert stats == {"total": 2, "upserted": 1, "deleted": 1, "unchanged": 1}
