"""会话冷归档与远期原话回查（第 2 步）的自动化测试。

覆盖：
1. JSONL 写入用户提问与回答（含拦截原因）、半截行容错；
2. 用户主动删除会话档案；
3. 5 轮回查只捞第 1 轮用户原话、绝不捞助手回答；
4. 没有明显指代且相关度不足时不注入；
5. 回查片段单片 ≤200 字节、合计 ≤500 字节。
"""
import asyncio

from local_rag import session_memory


def _user(round_no, content):
    return session_memory.archive_user_event(round_no, content)


def _answer(round_no, raw="助手回答占位", blocked=False):
    return session_memory.archive_answer_event(round_no, raw_answer=raw, answer=raw,
                                               blocked=blocked, blocking_issues=[{"type": "TEST"}])


def test_archive_writes_and_reads_user_and_answer_events(tmp_path):
    sid = "sess_archive_1"
    session_memory.append_event_sync(sid, _user(1, "我借给同事5万元，写了借条"), tmp_path)
    session_memory.append_event_sync(sid, _answer(1, "助手说要保存借条"), tmp_path)
    events = session_memory.read_events(sid, tmp_path)
    assert [event["role"] for event in events] == ["user", "assistant"]
    assert events[0]["content"] == "我借给同事5万元，写了借条"
    assert events[0]["round"] == events[0]["index"] == 1
    assert events[1]["raw_answer"] == "助手说要保存借条"
    assert events[1]["blocking_issues"] == [{"type": "TEST"}]
    assert events[1]["materials_sent"] == []
    assert isinstance(events[1]["timestamp"], float)
    # 回查数据源只包含用户原话。
    assert session_memory.read_user_turns(sid, tmp_path) == [
        {"round": 1, "content": "我借给同事5万元，写了借条"}]


def test_async_append_writes_through_thread(tmp_path):
    async def run():
        return await session_memory.append_event("sess_async", _user(1, "异步写入"), tmp_path)

    path = asyncio.run(run())
    assert path.exists()
    assert session_memory.read_user_turns("sess_async", tmp_path)[0]["content"] == "异步写入"


def test_archive_skips_truncated_tail_line(tmp_path):
    sid = "sess_half"
    path = session_memory.append_event_sync(sid, _user(1, "完整的一行"), tmp_path)
    # 模拟掉电：最后一行只写了一半，JSON 不闭合。
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"role":"user","content":"被截断的半行')
    events = session_memory.read_events(sid, tmp_path)
    assert len(events) == 1
    assert events[0]["role"] == "user"
    assert events[0]["content"] == "完整的一行"
    assert session_memory.read_user_turns(sid, tmp_path)[0]["content"] == "完整的一行"


def test_delete_session_archive_is_optional_and_idempotent(tmp_path):
    sid = "sess_delete"
    session_memory.append_event_sync(sid, _user(1, "需要被删除的记录"), tmp_path)
    assert session_memory.read_events(sid, tmp_path)
    assert session_memory.delete_session_archive(sid, tmp_path) is True
    assert session_memory.read_events(sid, tmp_path) == []
    assert session_memory.delete_session_archive(sid, tmp_path) is False


def test_recall_finds_first_turn_user_text_and_never_assistant_answer(tmp_path):
    sid = "sess_recall_5"
    rounds = [
        "2024年3月5日我借给同事5万元，写了借条，约定半年还",
        "他后来一直拖着不还",
        "我催过几次都没用",
        "现在想起诉但不知道准备什么",
    ]
    for index, text in enumerate(rounds, 1):
        session_memory.append_event_sync(sid, _user(index, text), tmp_path)
        session_memory.append_event_sync(
            sid, _answer(index, f"助手第{index}轮回答说：先保存借条和转账记录。"), tmp_path)
    turns = session_memory.read_user_turns(sid, tmp_path)
    hits = session_memory.recall("按照我最开始说的借条金额，这笔钱现在还能要回来吗", turns,
                                 current_round=5)
    assert hits, "第 5 轮应能回查到第 1 轮的借条金额"
    assert hits[0]["round"] == 1
    assert "5万元" in hits[0]["content"]
    assert hits[0]["line"].startswith("【前情陈述（第1轮用户原话）】：“")
    # 助手回答不在用户原话列表里，也不可能被回查注入。
    assert all("助手" not in turn["content"] for turn in turns)
    assert "助手" not in "".join(hit["line"] for hit in hits)


def test_recall_requires_cue_or_high_relevance():
    turns = [
        {"round": 1, "content": "今天天气不错"},
        {"round": 2, "content": "周末打算去公园"},
        {"round": 3, "content": "借条金额五万元还款期限是半年"},
    ]
    # 指代词存在，但和历史原话没有任何共同词项：不注入，留给“## 追问”。
    assert session_memory.recall("前面说的那个怎么办", turns, current_round=5) == []
    # 没有指代词，但本轮问句与第 3 轮原话高度重合：仍然回查。
    hits = session_memory.recall("借条金额五万元还款期限", turns, current_round=5)
    assert [hit["round"] for hit in hits] == [3]


def test_recall_piece_and_total_byte_limits():
    long_text = "借条金额五万元" * 80
    turns = [
        {"round": 1, "content": long_text},
        {"round": 2, "content": long_text + "补充说明"},
        {"round": 3, "content": "当前轮问题"},
    ]
    hits = session_memory.recall("最开始说的借条金额是多少", turns, current_round=4, limit=2)
    assert len(hits) == 2
    assert all(len(hit["line"].encode("utf-8")) <= 200 for hit in hits)
    assert sum(len(hit["line"].encode("utf-8")) for hit in hits) <= 500
    assert all("……" in hit["line"] for hit in hits)


def test_recall_ignores_current_and_immediately_previous_turn():
    turns = [
        {"round": 1, "content": "借条金额五万元"},
        {"round": 2, "content": "借条金额五万元，上一轮刚说过"},
        {"round": 3, "content": "借条金额五万元，本轮问题"},
    ]
    hits = session_memory.recall("按照最开始说的借条金额", turns, current_round=3)
    assert [hit["round"] for hit in hits] == [1]


def test_archive_path_rejects_unsafe_session_id(tmp_path):
    for sid in ("../escape", "a/b", "", "x" * 129):
        try:
            session_memory.archive_path(sid, tmp_path)
        except ValueError:
            continue
        raise AssertionError(f"不安全的会话编号未被拒绝：{sid!r}")
