"""检索诊断尺子回归：各路排名、归类优先级、分组汇总、快照对比与留出题保密规则。

只用合成数据和假检索器，不连 Qdrant/Ollama；真实运行由 scripts/measure_retrieval.py 另行留证。
"""
import asyncio
import json

import pytest

from local_rag.retrieval_measure import (
    build_key_index, compare_snapshots, gold_indices, first_rank, make_record, normalize_external,
    snapshot, summarize,
)
from scripts.measure_retrieval import exit_code, measure, write_run

ROWS = [dict(id=f"测试法/v1/{n}", title="测试法", locator=f"第{numeral}条 第1行", text=f"第{numeral}条 内容")
        for n, numeral in enumerate("一二三四五六七八九十", 1)]
QUESTION = dict(id="D0001", split="dev", category="测试类", query="测试问题甲乙",
                expect_any=[dict(title="测试法", article_no="第三条")])


def ranked_at(position, gold=2, score=1.0):
    """构造一路排名：正确条文排在第 position 名，前面用不相关的行号垫位。"""
    return [(1000 + n, 2.0) for n in range(position - 1)] + [(gold, score)]


def record(question=None, **changes):
    """用合成语料生成一题测量记录；默认各路都没找到，返回新字典。"""
    question = question or QUESTION
    args = dict(bm25_raw=[], bm25=[], dense=[], top4=[], top8=[], pool=[],
                ids=[row["id"] for row in ROWS], mode="hybrid_qdrant", expanded=False)
    args.update(changes)
    return make_record(question, gold_indices(question, build_key_index(ROWS)), **args)


def test_first_rank_counts_from_one_and_stops_at_depth():
    assert first_rank([5, 2, 7], {2}) == 2
    assert first_rank([5, 2], {9}) is None
    assert first_rank(list(range(100)), {59}, depth=60) == 60
    assert first_rank(list(range(100)), {60}, depth=60) is None


def test_gold_matches_title_and_article_number_in_any_numeral_style():
    index = build_key_index(ROWS + [dict(id="其他法/v1/3", title="其他法", locator="第三条")])
    assert gold_indices(QUESTION, index) == {2}
    arabic = dict(QUESTION, expect_any=[dict(title="测试法", article_no="第3条")])
    assert gold_indices(arabic, index) == {2}
    missing = dict(QUESTION, expect_any=[dict(title="不存在法", article_no="第三条")])
    assert gold_indices(missing, index) == set()


def test_hit_in_top_four_wins_over_every_other_category():
    hit = record(top4=[0, 2], top8=[0, 2], pool=[0, 2], bm25=ranked_at(1), dense=ranked_at(1, score=.30))
    assert hit["top4_rank"] == 2 and hit["diagnosis"] == "命中"
    assert hit["top4"] == ["测试法/v1/1", "测试法/v1/3"]


def test_dense_top20_below_threshold_outside_pool_is_threshold_block():
    blocked = record(dense=ranked_at(5, score=.30), bm25=ranked_at(30))
    assert blocked["dense_rank"] == 5 and blocked["dense_score"] == .30
    assert blocked["diagnosis"] == "被0.38挡掉"


def test_ranked_low_records_whether_answer_was_inside_pool():
    inside = record(pool=[0, 1, 3, 4, 5, 2], top4=[0, 1, 3, 4], top8=[0, 1, 3, 4, 5, 2])
    assert (inside["pool_rank"], inside["top8_rank"]) == (6, 6)
    assert inside["diagnosis"] == "排不上" and inside["outside_pool"] is False
    outside = record(bm25=ranked_at(35))
    assert outside["diagnosis"] == "排不上" and outside["outside_pool"] is True


def test_answer_beyond_sixty_is_not_found_and_errors_stay_separate():
    assert record(bm25=ranked_at(61), dense=ranked_at(61, score=.9))["diagnosis"] == "找不到"
    assert record(mode="error", error="TimeoutError")["diagnosis"] == "出错"


def test_summary_separates_splits_and_counts_routes_and_alias_effect():
    dev_hit = record(top4=[2], top8=[2], pool=[2], bm25=ranked_at(1), bm25_raw=ranked_at(9),
                     dense=ranked_at(3, score=.6), expanded=True)
    dev_low = record(dict(QUESTION, id="D0002"), bm25=ranked_at(15), bm25_raw=ranked_at(15),
                     dense=ranked_at(30, score=.5))
    hidden = record(dict(QUESTION, id="H0001", split="holdout"), bm25_raw=ranked_at(2), bm25=ranked_at(40),
                    expanded=True)
    summary = summarize([dev_hit, dev_low, hidden])
    dev = summary["dev"]
    assert (dev["count"], dev["top4_hits"], dev["top4_rate"]) == (2, 1, .5)
    assert dev["routes"]["bm25"] == {4: 1, 8: 1, 20: 2, 60: 2}
    assert dev["routes"]["bm25_raw"] == {4: 0, 8: 0, 20: 2, 60: 2}
    assert dev["routes"]["dense"] == {4: 1, 8: 1, 20: 1, 60: 2}
    assert dev["diagnosis"]["命中"] == 1 and dev["diagnosis"]["排不上"] == 1
    assert dev["alias_effect"] == dict(triggered=1, better=1, worse=0, same=0)
    assert dev["categories"]["测试类"] == dict(count=2, top4_hits=1)
    assert dev["mode_counts"] == {"hybrid_qdrant": 2}
    holdout = summary["holdout"]
    assert (holdout["count"], holdout["outside_pool"]) == (1, 1)
    assert holdout["alias_effect"] == dict(triggered=1, better=0, worse=1, same=0)


def test_display_hit_counts_answers_found_in_tail():
    shown = record(top8=[0, 1], tail=[5, 2])
    assert shown["tail_rank"] == 2 and shown["display_hit"] is True
    hidden = record(dict(QUESTION, id="D0002"), top8=[0, 1], tail=[5])
    assert hidden["tail_rank"] is None and hidden["display_hit"] is False
    assert record(top8=[2])["display_hit"] is True
    assert summarize([shown, hidden])["dev"]["display_hits"] == 1


def test_compare_snapshots_names_changed_ids_only_for_visible_splits():
    assert snapshot([record(top4=[2], top8=[2, 0])]) == {
        "D0001": dict(split="dev", top4=["测试法/v1/3"], top8=["测试法/v1/3", "测试法/v1/1"])}
    before = {"D1": dict(split="dev", top4=["a"], top8=["a", "b"]),
              "H1": dict(split="holdout", top4=["x"], top8=["x"])}
    after = {"D1": dict(split="dev", top4=["b"], top8=["a", "b"]),
             "H1": dict(split="holdout", top4=["y"], top8=["y"])}
    diff = compare_snapshots(before, after)
    assert diff["dev"] == dict(comparable=True, top4_changed=1, top8_changed=0,
                               top4_changed_ids=["D1"], top8_changed_ids=[])
    assert diff["holdout"] == dict(comparable=True, top4_changed=1, top8_changed=1)
    assert compare_snapshots(before, {"D1": after["D1"]})["holdout"]["comparable"] is False


def test_external_file_accepts_both_label_fields_and_errors_do_not_echo_questions():
    spec = {"questions": [
        dict(id="U1", query="老板不给钱", expect_any=[dict(title="测试法", article_no="第三条")]),
        dict(id="U2", query="房东不退押金", expect_hit_articles=[dict(title="测试法", article_no="第四条", version_id="v1")]),
    ]}
    questions = normalize_external(spec)
    assert [q["split"] for q in questions] == ["external", "external"]
    assert questions[1]["expect_any"] == [dict(title="测试法", article_no="第四条")]
    assert normalize_external(spec["questions"]) == questions
    with pytest.raises(ValueError) as error:
        normalize_external([dict(id="U3", query="邻居家的狗咬了我")])
    assert "第1题" in str(error.value) and "邻居家的狗" not in str(error.value)


def test_written_run_keeps_hidden_questions_out_of_every_file(tmp_path):
    secret = dict(QUESTION, id="H0001", split="holdout", query="不能外泄的留出题原文",
                  expect_any=[dict(title="保密法", article_no="第三条")])
    own = dict(QUESTION, id="U1", split="external", query="用户自己写的题")
    records = [record(), record(secret), record(own)]
    directory = write_run(tmp_path, dict(splits=summarize(records)), records, snapshot(records))
    details = json.loads((directory / "details.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in details] == ["D0001"]
    assert json.loads((directory / "summary.json").read_text(encoding="utf-8"))["splits"]["holdout"]["count"] == 1
    for path in directory.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "不能外泄" not in text and "用户自己写的题" not in text and "保密法" not in text


def test_runs_started_in_the_same_instant_get_separate_directories(tmp_path):
    first = write_run(tmp_path, {}, [], {}, stamp="20261102-120000-000000")
    second = write_run(tmp_path, {}, [], {}, stamp="20261102-120000-000000")
    assert first != second and (first / "summary.json").exists() and (second / "summary.json").exists()


class FakeBM25:
    """按固定次序返回词面排名，并记录收到的查询文本。"""
    def __init__(self, ranking):
        self.ranking, self.queries = ranking, []

    def rank(self, query):
        self.queries.append(query)
        return list(self.ranking)


class FakeStore:
    """模拟只读 Qdrant：记录每次请求体，固定返回给定的语义候选。"""
    collection, ready, qualified_count = "legal_rag_test", True, 0

    def __init__(self, points):
        self.points, self.bodies = points, []

    def scope(self):
        return {"must": [{"key": "corpus_id", "match": {"value": "test"}}]}

    async def request(self, method, path, body=None):
        self.bodies.append((method, path, body))
        return {"points": self.points}


class FakeEmbedder:
    """返回固定二维向量，代替本机 embedding 服务。"""
    async def embed(self, texts, query=False):
        return [[1.0, 0.0] for _ in texts]


class FakeRetriever:
    """只实现测量要用到的生产检索器接口；search 按固定融合次序截取 top_k，尾部按给定次序返回。"""
    def __init__(self, rows, bm25, store, fused, tail=()):
        self.rows, self._bm25, self.store, self.fused, self.tail = rows, bm25, store, fused, list(tail)
        self.embedder, self.vectors, self.tail_calls = FakeEmbedder(), [[1.0, 0.0]], []

    async def search(self, query, top_k=4):
        self.store.qualified_count = 1
        return [dict(self.rows[i], retrieval_score=1.0) for i in self.fused[:top_k]], "hybrid_qdrant"

    async def tail_candidates(self, query, exclude_ids, depth=60):
        self.tail_calls.append((query, set(exclude_ids), depth))
        return [dict(self.rows[i]) for i in self.tail]


def test_measure_collects_route_ranks_and_queries_dense_without_threshold():
    store = FakeStore([{"score": .41, "payload": {"chunk_id": ROWS[5]["id"]}},
                       {"score": .33, "payload": {"chunk_id": ROWS[2]["id"]}}])
    retriever = FakeRetriever(ROWS, FakeBM25([(4, 3.0), (2, 2.0)]), store, fused=[7, 2, 1], tail=[9, 2])
    [result] = asyncio.run(measure(retriever, [QUESTION], depth=60))
    assert (result["bm25_raw_rank"], result["bm25_rank"], result["dense_rank"], result["dense_score"]) == (2, 2, 2, .33)
    assert (result["top4_rank"], result["pool_rank"], result["mode"], result["diagnosis"]) == (2, 2, "hybrid_qdrant", "命中")
    assert result["tail_rank"] == 2
    assert retriever.tail_calls == [("测试问题甲乙", {ROWS[7]["id"], ROWS[2]["id"], ROWS[1]["id"]}, 60)]
    method, path, body = store.bodies[-1]
    assert (method, path) == ("POST", "/collections/legal_rag_test/points/query")
    assert body["limit"] == 60 and "score_threshold" not in body


def test_exit_code_flags_degraded_errored_or_changed_runs():
    assert exit_code([record()]) == 0
    assert exit_code([record(mode="lexical_no_qualified_dense")]) == 2
    assert exit_code([record(mode="error", error="TimeoutError")]) == 2
    assert exit_code([record()], source_changed=True) == 2


def test_alias_ablation_restores_production_table_even_when_measurement_fails():
    """关闭词表的试验不能泄漏到后续测量；真实扩展入口也必须随之关闭。"""
    import local_rag.retrieval as retrieval
    from scripts.measure_retrieval import alias_ablation

    original = retrieval.COLLOQUIAL_ALIASES
    query = next(iter(original))
    expected = retrieval.expand_colloquial_query(query)
    assert expected != query
    with pytest.raises(RuntimeError, match="测量中断"):
        with alias_ablation(True):
            assert retrieval.COLLOQUIAL_ALIASES == {}
            assert retrieval.expand_colloquial_query(query) == query
            raise RuntimeError("测量中断")
    assert retrieval.COLLOQUIAL_ALIASES is original
    assert retrieval.expand_colloquial_query(query) == expected
    with alias_ablation(False):
        assert retrieval.COLLOQUIAL_ALIASES is original
