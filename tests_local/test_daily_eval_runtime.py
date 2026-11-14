"""评测运行边界：禁止写库、如实标记无向量降级、冻结题集和完整材料观测。"""
import asyncio
import json

import httpx
import pytest

from local_rag.daily_eval_runtime import ReadOnlyQdrant, actual_search, load_dataset, prompt_materials, save_report
from local_rag.retrieval import ROOT


def test_readonly_qdrant_rejects_writes_before_transport():
    calls = []
    def handle(request):
        calls.append(request.method)
        return httpx.Response(200, json={"result": {"count": 1}})
    async def run():
        store = ReadOnlyQdrant(transport=httpx.MockTransport(handle))
        try:
            assert await store.request("POST", "/collections/legal/points/count", {}) == {"count": 1}
            with pytest.raises(RuntimeError, match="禁止写"):
                await store.request("PUT", "/collections/legal/points", {})
            with pytest.raises(RuntimeError, match="禁止写"):
                await store.request("POST", "/collections/legal/points/delete", {})
        finally:
            await store.close()
    asyncio.run(run())
    assert calls == ["POST"]


def test_empty_qualified_dense_is_not_reported_as_hybrid():
    class Retriever:
        class Store:
            qualified_count = 0
        store = Store()
        async def search(self, query, top_k=4):
            return [{"title": "法", "locator": "第一条"}], "hybrid_qdrant"
    assert asyncio.run(actual_search(Retriever(), "问句"))[1] == "lexical_no_qualified_dense"


@pytest.mark.parametrize('top_k', [4, 5, 6, 8])
def test_answer_observer_preserves_requested_candidate_count(top_k):
    """回答评测必须透传主检索/展示条数，不能把5/6条试验暗中截成4条。"""
    from scripts.eval_answers import ObservedRetriever

    class Retriever:
        rows = []
        class Store:
            qualified_count = 0
        store = Store()

        async def search(self, query, top_k=4):
            self.store.qualified_count = top_k
            self.called = (query, top_k)
            return [{'id': str(i)} for i in range(top_k)], 'hybrid_qdrant'

    inner = Retriever()
    hits, mode = asyncio.run(ObservedRetriever(inner).search('普通问题', top_k=top_k))
    assert inner.called == ('普通问题', top_k)
    assert len(hits) == top_k and mode == 'hybrid_qdrant'


def test_dataset_validates_all_real_labels_and_protects_old_files():
    from local_rag.retrieval import load_corpus
    from local_rag.daily_evaluation import fixed_split
    rows, _ = load_corpus()
    dataset, digest = load_dataset(ROOT / "data/eval/daily_eval_v3.json", rows)
    assert len(digest) == 64 and len(dataset["questions"]) == 120
    assert dataset["questions"] == fixed_split(dataset["questions"], dataset["seed"])
    assert len({q["category"] for q in dataset["questions"]}) == 12
    assert all(q["label_status"] == "AI草拟待人工抽检" for q in dataset["questions"])


def test_report_is_unique_and_explicit_about_degraded_mode(tmp_path):
    report = dict(kind="retrieval", scope={"dataset_sha256": "a", "corpus_id": "b", "embedding_id": "c",
                                          "mode_counts": {"lexical": 1}},
                  results=[dict(id="D1", split="holdout", category="消费", query="问句", hit=False,
                                reciprocal_rank=0, mode="lexical", expect_any=[], top_4=[], error=None)],
                  summary={}, degradation_reason="服务未启动", elapsed_seconds=1)
    first = save_report(report, tmp_path)
    second = save_report(report, tmp_path)
    assert first != second
    assert (first / "retrieval.json").exists()
    text = (second / "report.md").read_text(encoding="utf-8")
    assert "降级" in text and "服务未启动" in text and "holdout" in text
    assert json.loads((second / "retrieval.json").read_text(encoding="utf-8"))["comparison"]["comparable"]


def test_save_report_same_clock_tick_creates_independent_comparable_reports(monkeypatch, tmp_path):
    """固定同一时钟刻度：撞名必须递增编号，不能覆盖旧JSON/报告或丢失比较来源。"""
    from datetime import datetime
    import local_rag.daily_eval_runtime as runtime

    class FrozenDatetime:
        @staticmethod
        def now():
            return datetime(2026, 11, 3, 12, 0, 0, 123456)

    monkeypatch.setattr(runtime, 'datetime', FrozenDatetime)
    report = dict(kind='retrieval', scope=dict(dataset_sha256='a', corpus_id='b', embedding_id='c',
                                              mode_counts={'lexical': 1}),
                  results=[dict(id='D1', split='dev', category='测试', query='测试问题', hit=False,
                                reciprocal_rank=0, mode='lexical', expect_any=[], top_4=[], error=None)],
                  summary={}, degradation_reason='服务未启动', elapsed_seconds=1)
    first = runtime.save_report(dict(report, label='first'), tmp_path)
    second = runtime.save_report(dict(report, label='second'), tmp_path)
    assert first.name == '20261103-120000-123456'
    assert second.name == '20261103-120000-123456-2'
    for directory, label in ((first, 'first'), (second, 'second')):
        saved = json.loads((directory / 'retrieval.json').read_text(encoding='utf-8'))
        assert saved['label'] == label and saved['run_directory'] == str(directory)
        assert '服务未启动' in (directory / 'report.md').read_text(encoding='utf-8')
    saved = json.loads((second / 'retrieval.json').read_text(encoding='utf-8'))
    assert saved['previous_report'] == str(first / 'retrieval.json')
    assert saved['comparison']['comparable'] is True


def test_error_evidence_recovers_only_whole_materials_from_actual_model_prompt():
    from prototype.rag.input_builder import Material, RagInput
    material = Material(material_id="M1", kind="LAW_ARTICLE", title="测试法", locator="第二条 行3",
                        version_id="2026", source_url="https://www.gov.cn/x", text="第二条 应当退款。\n另一款。",
                        verified_scope="测试范围")
    prompt = RagInput(question="问题", materials=[material], gaps=["缺口"]).render_user_prompt()
    recovered = prompt_materials(prompt)
    assert len(recovered) == 1
    assert recovered[0]["material_id"] == "M1" and recovered[0]["text"] == material.text
    assert recovered[0]["title"] == material.title
    assert prompt_materials("【检索材料】\n（本次无检索材料）") == []


def test_wrong_live_corpus_fingerprint_downgrades_without_query_or_write(monkeypatch):
    """Web声称ready也不能代替本地指纹核对；不匹配时不发向量查询，更不允许重建索引。"""
    import local_rag.daily_eval_runtime as runtime
    calls = []
    row = dict(id="r", title="测试法", text="退款", locator="第一条")
    monkeypatch.setattr(runtime, "load_corpus", lambda: ([row], {"chunks": 1}))
    client_class = httpx.AsyncClient
    def handle(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/status":
            return httpx.Response(200, json=dict(index={"state": "ready"}, model="qwen3:4b", embedding="embed:local",
                database=dict(ready=True, collection="legal_rag_test", corpus_id="0" * 64, points=1)))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [dict(name="embed:local", digest="e"), dict(name="qwen3:4b", digest="m")]})
        raise AssertionError("指纹不符仍然访问了数据库")
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs:
                        client_class(*args, **{**kwargs, "transport": httpx.MockTransport(handle)}))
    async def run():
        async with runtime.open_runtime() as (retriever, _, info):
            assert retriever.vectors is None
            assert "指纹" in info["degradation_reason"]
            assert (await runtime.actual_search(retriever, "退款"))[1] == "lexical"
    asyncio.run(run())
    assert calls == [("GET", "/api/status"), ("GET", "/api/tags")]


def test_frozen_dataset_rejects_rewording_before_scoring(tmp_path):
    from local_rag.retrieval import load_corpus
    rows, _ = load_corpus()
    original = ROOT / "data/eval/daily_eval_v3.json"
    spec = json.loads(original.read_text(encoding="utf-8"))
    spec["protected_eval_sha256"] = {}
    spec["questions"][0]["query"] += "请再说一遍"
    target = tmp_path / original.name
    target.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    target.with_suffix(".frozen.json").write_bytes(original.with_suffix(".frozen.json").read_bytes())
    with pytest.raises(ValueError, match="冻结题集哈希"):
        load_dataset(target, rows)
