"""远期原话回查接入问答流程的回归测试（多轮会话底座第 2 步）。

覆盖：
1. 5 轮对话里第 5 轮按“最开始说的借条金额”回查第 1 轮用户原话，并注入本轮用户消息；
2. 回查只在用户原话上做，助手回答不会进入注入内容；
3. 当轮注入的远期原话会并入金额/期限校验器的豁免上下文，避免 5 万元被误判捏造；
4. 单轮没有历史时回查为空，校验上下文与本轮问题完全一致。
"""
import asyncio

import local_rag.service as service_module
from local_rag import session_memory
from local_rag.service import ChatService, amount_grounding_issues, deadline_grounding_issues
from tests_local.test_answer_permission import reading


class Search:
    """空检索桩：本组只验证会话记忆与校验上下文，不依赖法规库。"""
    async def search(self, query, top_k=4):
        return [], "lexical"


class CaptureModel:
    """记录每次送模消息的替身；回答使用合法的材料阅读格式。"""
    def __init__(self):
        self.calls = []

    async def chat(self, messages):
        self.calls.append(messages)
        return reading(quote=None)


ROUNDS = [
    "2024年3月5日我借给同事5万元，写了借条，约定半年还",
    "他后来一直拖着不还",
    "我催过几次都没用",
    "现在想起诉但不知道准备什么",
    "按照我最开始说的借条金额，这笔钱现在还能要回来吗",
]


def _run_rounds(service, sid, rounds):
    async def run():
        for question in rounds:
            rid = service.start(sid, question)
            await service.sessions[sid].jobs[rid]["task"]
            job = service.result(sid, rid)
            assert job["status"] == "done", (question, job)
        return True

    return asyncio.run(run())


def test_fifth_round_injects_first_round_user_statement(tmp_path):
    model = CaptureModel()
    service = ChatService(Search(), model, official_catalog=[], session_dir=tmp_path)
    sid = service.create()
    _run_rounds(service, sid, ROUNDS)

    final_user = model.calls[-1][-1]["content"]
    assert "【前情陈述（第1轮用户原话）】：“" in final_user
    assert "5万元" in final_user
    # 前情陈述只能在本轮用户消息里，不能混进 system。
    assert "前情陈述" not in model.calls[-1][0]["content"]
    assert final_user.count("【前情陈述（第1轮用户原话）】") == 1

    events = session_memory.read_events(sid, tmp_path)
    assert [event["role"] for event in events] == ["user", "assistant"] * 5
    assert events[0]["content"] == ROUNDS[0]
    assert events[1]["raw_answer"]
    assert session_memory.read_user_turns(sid, tmp_path)[0]["content"] == ROUNDS[0]


def test_assistant_answer_is_never_in_recall_source(tmp_path):
    model = CaptureModel()
    service = ChatService(Search(), model, official_catalog=[], session_dir=tmp_path)
    sid = service.create()
    _run_rounds(service, sid, ROUNDS)
    turns = session_memory.read_user_turns(sid, tmp_path)
    hits = session_memory.recall(ROUNDS[-1], turns, current_round=5)
    assert [hit["round"] for hit in hits] == [1]
    # 助手事件里的 raw_answer/answer 不会被 read_user_turns 当成候选。
    assistant_texts = [event.get("raw_answer", "") for event in session_memory.read_events(sid, tmp_path)
                       if event["role"] == "assistant"]
    assert assistant_texts and not any(text in hit["line"] for hit in hits for text in assistant_texts)


def test_recalled_amount_exempts_user_reference_from_grounding():
    """修复前后对照：同一句“支付5万元”，只认本轮问题时被拦，带上第 1 轮原话后放行。"""
    materials = [dict(material_id="M1", text="第一条 当事人应当保存与纠纷有关的原始凭证。")]
    answer = "按您最开始说的借款金额，对方应支付5万元。"
    before = amount_grounding_issues(answer, materials, question="那笔钱现在还能要回来吗")
    after = amount_grounding_issues(
        answer, materials, question="那笔钱现在还能要回来吗 2024年3月5日我借给同事5万元，写了借条")
    assert before and before[0]["type"] == "AMOUNT_VALUE_NOT_IN_CITED_TEXT"
    assert after == []


def test_recalled_deadline_exempts_user_reference_from_grounding():
    materials = [dict(material_id="M1", text="第一条 当事人应当保存与纠纷有关的原始凭证。")]
    answer = "应当在30日内提出。"
    before = deadline_grounding_issues(answer, materials, question="多久内要提出")
    after = deadline_grounding_issues(answer, materials, question="多久内要提出 对方当时说30日内答复")
    assert before and before[0]["type"] == "DEADLINE_VALUE_NOT_IN_CITED_TEXT"
    assert after == []


def test_service_passes_recalled_text_into_both_grounding_checks(monkeypatch, tmp_path):
    captured = {"amount": [], "deadline": []}
    real_amount = service_module.amount_grounding_issues
    real_deadline = service_module.deadline_grounding_issues

    def spy_amount(answer, materials, *, question=""):
        captured["amount"].append(question)
        return real_amount(answer, materials, question=question)

    def spy_deadline(answer, materials, *, question=""):
        captured["deadline"].append(question)
        return real_deadline(answer, materials, question=question)

    monkeypatch.setattr(service_module, "amount_grounding_issues", spy_amount)
    monkeypatch.setattr(service_module, "deadline_grounding_issues", spy_deadline)
    model = CaptureModel()
    service = ChatService(Search(), model, official_catalog=[], session_dir=tmp_path)
    sid = service.create()
    _run_rounds(service, sid, ROUNDS)

    # 前四轮没有可用的远期原话，上下文就是本轮问题；第五轮才并入第 1 轮原话。
    assert captured["amount"][:4] == ROUNDS[:4]
    assert captured["deadline"][:4] == ROUNDS[:4]
    assert captured["amount"][4].startswith(ROUNDS[4])
    assert "5万元" in captured["amount"][4]
    assert captured["deadline"][4].startswith(ROUNDS[4])
    assert "5万元" in captured["deadline"][4]


def test_single_turn_has_no_recall_and_keeps_question_unchanged(monkeypatch, tmp_path):
    captured = []
    real_amount = service_module.amount_grounding_issues

    def spy_amount(answer, materials, *, question=""):
        captured.append(question)
        return real_amount(answer, materials, question=question)

    monkeypatch.setattr(service_module, "amount_grounding_issues", spy_amount)
    model = CaptureModel()
    service = ChatService(Search(), model, official_catalog=[], session_dir=tmp_path)
    sid = service.create()
    question = "公司辞退我不给工资怎么办"
    _run_rounds(service, sid, [question])
    assert captured == [question]
    assert "前情陈述" not in str(model.calls[0])
