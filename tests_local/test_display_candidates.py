"""展示候选回归：尾部候选的取法与失败处理、问答结果的 display 字段、页面"更多候选"渲染。

只用合成语料和替身，不连 Qdrant/Ollama；真实效果由 scripts/measure_retrieval.py 另行测量。
"""
import asyncio

import pytest

from local_rag.retrieval import Retriever, expand_colloquial_query
from local_rag.service import ChatService
from tests_local.test_official_npc_live import _render_sources


def article(i, **extra):
    """生成第 i 条合成条文；返回新字典，不代表真实法规。"""
    return dict(id=f"R{i}", kind="LAW_ARTICLE", title=f"规范{i}", locator=f"第{i + 1}条", version_id="测试版本",
                text=f"第{i + 1}条 内容{i}。", source_url="https://www.gov.cn/", verified_scope="测试替身",
                content_scope="ARTICLES_ONLY", **extra)


ROWS = [article(i) for i in range(100)]


class Embedder:
    """记录问题向量化的调用，所有文本都返回同一个二维向量。"""
    model_id = "fake-embedding:1"

    def __init__(self):
        self.calls = []

    async def embed(self, texts, *, query=False):
        self.calls.append((list(texts), query))
        return [[0.0, 1.0] for _ in texts]


class DenseStore:
    """模拟线上语义检索：记录每次要的条数，按给定次序返回已过0.38阈值的候选，可模拟故障。"""
    ready = True

    def __init__(self, hits=(), error=None):
        self.hits, self.error, self.limits = list(hits), error, []

    async def search(self, vector, limit):
        self.limits.append(limit)
        if self.error:
            raise self.error
        return self.hits[:limit]


def make_retriever(monkeypatch, bm25_order, store):
    """用合成语料建检索器：关键词排名按 bm25_order 固定，语义走给定的 store。"""
    retriever = Retriever(ROWS, Embedder())
    retriever.vectors = [[0.0, 1.0]]
    retriever.store = store
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(i, 10.0) for i in bm25_order])
    return retriever


def test_tail_merges_both_routes_top60_minus_excluded_by_best_rank(monkeypatch):
    store = DenseStore([(50, .9), (0, .8), (51, .7)])
    retriever = make_retriever(monkeypatch, list(range(100)), store)
    tail = asyncio.run(retriever.tail_candidates("普通问题", {"R0"}))
    ids = [row["id"] for row in tail]
    assert ids[:5] == ["R50", "R1", "R51", "R2", "R3"]
    assert "R0" not in ids and "R59" in ids and "R60" not in ids
    assert store.limits == [60]
    assert tail[0]["tail_routes"] == {"bm25": 51, "dense": 1}
    assert tail[1]["tail_routes"] == {"bm25": 2, "dense": None}


def test_tail_keyword_route_uses_expanded_query_and_dense_uses_original(monkeypatch):
    seen = []
    retriever = make_retriever(monkeypatch, [], DenseStore())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: seen.append(query) or [])
    asyncio.run(retriever.tail_candidates("老板不发工资", set()))
    assert seen == [expand_colloquial_query("老板不发工资")] and seen[0] != "老板不发工资"
    assert retriever.embedder.calls[-1] == (["老板不发工资"], True)


def test_tail_dense_failure_keeps_keyword_part_and_database_state(monkeypatch):
    store = DenseStore(error=RuntimeError("Qdrant 不可用"))
    retriever = make_retriever(monkeypatch, [3, 4], store)
    retriever.state, retriever.error = "ready", ""
    tail = asyncio.run(retriever.tail_candidates("普通问题", set()))
    assert [row["id"] for row in tail] == ["R3", "R4"]
    assert store.ready is True and retriever.state == "ready" and retriever.error == ""


def test_tail_without_vectors_is_keyword_only(monkeypatch):
    retriever = make_retriever(monkeypatch, [7], DenseStore([(8, .9)]))
    retriever.vectors = None
    tail = asyncio.run(retriever.tail_candidates("普通问题", set()))
    assert [row["id"] for row in tail] == ["R7"] and retriever.embedder.calls == []


class DisplaySearch:
    """按 top_k 截取固定次序，并给出两条尾部候选；记录每次检索收到的 top_k。"""
    rows = [article(i) for i in range(12)]

    def __init__(self):
        self.calls = []

    async def search(self, query, top_k=4):
        self.calls.append(top_k)
        return [dict(row) for row in self.rows[:top_k]], "lexical"

    async def tail_candidates(self, query, exclude_ids, depth=60):
        return [dict(row, tail_routes={"bm25": None, "dense": 9 + n})
                for n, row in enumerate(self.rows[8:10]) if row["id"] not in exclude_ids]


class Answer:
    """固定的合规回答，不引用任何材料。"""
    async def chat(self, messages):
        return "## 追问\n无。\n## 处理方向\n先核对材料。\n## 证据清单\n保存记录。\n## 不确定性\n仍需核实。\nUSED_IDS: 无"


def run_question(retriever, question="工资被拖欠怎么办"):
    """在关闭官方目录的内存服务里问一个问题，返回任务结果。"""
    async def run():
        service = ChatService(retriever, Answer(), official_catalog=[])
        sid = service.create()
        rid = service.start(sid, question)
        await service.sessions[sid].jobs[rid]["task"]
        return service.result(sid, rid)
    return asyncio.run(run())


def test_result_lists_top8_with_sent_flags_and_tail_without_changing_model_input():
    retriever = DisplaySearch()
    job = run_question(retriever)
    assert job["status"] == "done", job
    result = job["result"]
    top, tail = result["display"]["top"], result["display"]["tail"]
    assert [row["id"] for row in top] == [f"R{i}" for i in range(8)]
    assert [row["sent_to_model"] for row in top] == [True] * 4 + [False] * 4
    assert [row["material_id"] for row in top[:5]] == ["M1", "M2", "M3", "M4", None]
    assert [row["id"] for row in tail] == ["R8", "R9"] and tail[0]["routes"] == {"bm25": None, "dense": 9}
    assert not any(row["sent_to_model"] for row in tail)
    assert [m["title"] for m in result["retrieved"]] == [f"规范{i}" for i in range(4)]
    assert retriever.calls == [4, 8]


def test_display_failure_does_not_block_the_answer():
    class BrokenDisplay(DisplaySearch):
        async def search(self, query, top_k=4):
            if top_k == 8:
                raise RuntimeError("展示检索失败")
            return await super().search(query, top_k)

    class OldSearch:
        async def search(self, query):
            return [article(i) for i in range(4)], "lexical"

    for retriever in (BrokenDisplay(), OldSearch()):
        job = run_question(retriever)
        assert job["status"] == "done", job
        assert job["result"]["display"] == {"top": [], "tail": []}
        assert len(job["result"]["retrieved"]) == 4


def test_display_dense_failure_keeps_main_search_state(monkeypatch):
    """使用真实 search 的内部降级路径，展示查询失败不能覆盖主检索的成功状态。"""
    async def run():
        store = DenseStore([(3, .9)])
        retriever = make_retriever(monkeypatch, [3, 4], store)
        retriever.state, retriever.error = 'ready', ''
        main, mode = await retriever.search('普通问题')
        assert main and mode == 'hybrid_qdrant'
        store.error = TimeoutError('模拟展示查询超时')
        service = ChatService(retriever, Answer(), official_catalog=[])
        top, _ = await service._display_candidates('普通问题')
        assert top  # 内部降级仍可展示关键词候选。
        assert store.ready is True
        assert (retriever.state, retriever.error) == ('ready', '')
        store.error = None
        recovered, mode = await retriever.search('普通问题')
        assert recovered == main and mode == 'hybrid_qdrant'
        assert (retriever.state, retriever.error) == ('ready', '')

    asyncio.run(run())


@pytest.mark.parametrize('failure', [RuntimeError, asyncio.CancelledError])
def test_display_exception_preserves_main_search_state(failure):
    """外抛异常或取消也必须保留主检索状态；取消继续向上传播。"""
    class BrokenDisplay(DisplaySearch):
        state, error = 'ready', ''

        async def search(self, query, top_k=4):
            self.state, self.error = 'lexical', '展示查询失败'
            raise failure('模拟展示查询中断')

    retriever = BrokenDisplay()
    service = ChatService(retriever, Answer(), official_catalog=[])
    if failure is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(service._display_candidates('普通问题'))
    else:
        assert asyncio.run(service._display_candidates('普通问题')) == ([], [])
    assert (retriever.state, retriever.error) == ('ready', '')


def test_display_success_preserves_main_search_degradation(monkeypatch):
    """展示查询偶然成功也不能把本轮主检索的降级记录抹掉。"""
    retriever = make_retriever(monkeypatch, [3], DenseStore([(3, .9)]))
    retriever.state, retriever.error = 'lexical', '主检索暂不可用'
    service = ChatService(retriever, Answer(), official_catalog=[])
    top, _ = asyncio.run(service._display_candidates('普通问题'))
    assert top
    assert (retriever.state, retriever.error) == ('lexical', '主检索暂不可用')


def test_scope_degraded_question_shows_no_candidates():
    question = "我想申请上海市徐汇区2026年人才租房补贴，请按该区当年的实施细则告诉我收入门槛和补贴金额。"
    job = run_question(DisplaySearch(), question)
    assert job["status"] == "done", job
    assert job["result"]["scope_degraded"] and job["result"]["display"] == {"top": [], "tail": []}


def test_candidate_view_marks_sent_rows_and_collapses_tail():
    result = dict(citations=[], retrieved=[article(0, material_id="M1")], official_fallback=[], display=dict(
        top=[article(0, material_id="M1", sent_to_model=True), article(1, material_id=None, sent_to_model=False)],
        tail=[article(9, material_id=None, sent_to_model=False, routes={"bm25": 22, "dense": 15})]))
    rendered = _render_sources(result, True)
    text = rendered["text"]
    assert "M1 · 已送入模型" in text and "未送入模型" in text
    assert "更多候选（1条" in text and "语义第15名" in text and "关键词第22名" in text
    tails = [child for child in rendered["children"] if child["tag"] == "details"]
    assert len(tails) == 1 and "规范9" in tails[0]["text"]


def test_candidate_view_without_display_keeps_old_behaviour():
    rendered = _render_sources(dict(citations=[], retrieved=[article(0, material_id="M1")], official_fallback=[]), True)
    assert "规范0" in rendered["text"] and "更多候选" not in rendered["text"]
