import asyncio

from backend.pipeline import query


def test_geography_expansion_timeout_falls_back_to_original_terms(monkeypatch):
    class TimeoutLlm:
        model_name = "test"

        @staticmethod
        async def ainvoke(_prompt):
            raise asyncio.TimeoutError

    class Tracker:
        def add_usage(self, *_args, **_kwargs):
            raise AssertionError("Timed-out calls must not record usage")

    monkeypatch.setattr(query, "llm", TimeoutLlm())
    result = asyncio.run(query._expand_geographies_with_llm(["North America"], Tracker()))
    assert result == []
