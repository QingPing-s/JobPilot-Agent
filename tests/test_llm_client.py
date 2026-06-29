from types import SimpleNamespace

from src.llm_client import DeepSeekClient, LLMConfig, get_token_usage, reset_token_usage


class FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5, total_tokens=17),
        )


def test_deepseek_client_records_token_usage(monkeypatch):
    monkeypatch.setenv("LLM_INPUT_COST_PER_MILLION", "1")
    monkeypatch.setenv("LLM_OUTPUT_COST_PER_MILLION", "2")
    reset_token_usage()
    client = DeepSeekClient(LLMConfig(api_key="test-key"))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    content = client.chat(messages=[{"role": "user", "content": "hello"}])

    assert content == "ok"
    assert get_token_usage() == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
        "calls": 1,
        "estimated_cost_usd": 0.000022,
    }
