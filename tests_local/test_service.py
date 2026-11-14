"""问答核心服务（ChatService）与模型适配器自动化测试

用途说明：
验证 local_rag/service.py 中的核心调度逻辑，包含：
1. 本地连接白名单校验（拒绝任何非本机外部 Ollama 地址，防数据外泄）。
2. 用户主动取消任务时，在途任务必须优雅中断，绝不将半截回答写入历史。
3. 会话重置（Reset）时，旧会话彻底销毁，异步晚返回的结果绝不能写回新会话。
4. 本地推理超时保护。
5. 内存防爆炸限制（会话总数上限、过期自动清理）；多轮不再按轮数硬拒绝。
6. 未收录地方补贴细则的范围降级、地域标题边界与共享判分校验。
7. 分步实操、追问编号、多主体双向责任及完整引用声明的实际传递，以及整条材料输入预算。
8. 校验后的法名补全、展示编号清理与原始审计/引用卡片保留。
9. 候选材料场景与主体的引用前自检要求、无直接材料时的通用核实与无编号声明。
10. 分场景实操、街头刑事应对及无关机构/员工手册禁令，保留原有上下文预算。
11. 句尾声明的原始审计与展示去除，保留标点、空声明后的正文及三张引用卡片。
12. 提示词预算：材料第一顺位保底、历史按余量倒序填补、单轮逐字节钉扎与 12 轮抗压。
"""
import asyncio
from dataclasses import replace
import json
import pathlib
import pytest
import local_rag.service as service_module
from local_rag.service import ChatService, SessionError, uncovered_foreign_region
from local_rag.ollama import OllamaClient
from local_rag.evaluation import judge_degrade
from prototype.rag.input_builder import Material
from prototype.rag.citation_check import parse_used_ids
from prototype.rag.postprocess import postprocess
from tests_local.test_answer_permission import reading


class Search:
    """测试用检索桩（Mock Retriever）：快速返回空结果，隔离数据库依赖"""
    async def search(self, query, top_k=5):
        return [], "lexical"


class FakeModel:
    """可控异步模型桩：通过 asyncio.Event 控制模型响应时机，专用于测试取消与并发晚返回"""
    def __init__(self):
        self.messages = None
        self.gate = asyncio.Event()
        self.cancelled = False

    async def chat(self, messages):
        self.messages = messages
        try:
            # 等待测试用例手动放行，方便制造正在思考中被取消的场景
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return reading(quote=None)


@pytest.fixture
def display_materials():
    """合成材料只验证引用身份与展示，不作为真实法规快照或法律适用结论。"""
    return [Material(material_id=mid, kind="LAW_ARTICLE", title=title, locator=article,
                     version_id="测试版本", source_url="https://www.gov.cn/", text=text,
                     verified_scope="测试替身", content_scope="ARTICLES_ONLY")
            for mid, title, article, text in [
                ("M1", "中华人民共和国教育法", "第四十三条", "第四十三条 受教育者享有提出申诉的权利。"),
                ("M2", "中华人民共和国高等教育法", "第五十八条", "第五十八条 学业成绩合格，准予毕业。"),
            ]]


@pytest.mark.parametrize("raw, expected", [
    ("依据[M1]第四十三条提出申诉[M1][M2]。\nUSED_IDS: M1, M2", "依据《中华人民共和国教育法》第四十三条提出申诉。"),
    ("依据《中华人民共和国教育法》[M1]第四十三条处理。", "依据《中华人民共和国教育法》第四十三条处理。"),
    ("依据《中华人民共和国教育法》 [M1]第四十三条处理。", "依据《中华人民共和国教育法》 第四十三条处理。"),
    ("上一段提到《其他规范》\n[M1]第四十三条可核对。", "上一段提到《其他规范》\n《中华人民共和国教育法》第四十三条可核对。"),
    ("依据M 2 第58条核对条件。", "依据《中华人民共和国高等教育法》第58条核对条件。"),
    ("保留通知（M1），核对条件(M2)[ M 1 ] 。\nUSED_IDS：M1, M2", "保留通知，核对条件。"),
    ("依据[M99]第四十三条说明。", "依据第四十三条说明。"),
    ("型号AM1第四十三条不应被改写。", "型号AM1第四十三条不应被改写。"),
    ("## 处理方向\n[M1]先保留通知。\n（M2）再核对条件。", "## 处理方向\n先保留通知。\n再核对条件。"),
    ("正文结论。USED_IDS: M1 M2 M3", "正文结论。"),
    ("## 不确定性\nUSED_IDS:\n尚需核对书面通知。", "## 不确定性\n\n尚需核对书面通知。"),
    (None, ""), ("", ""),
])
def test_clean_display_answer_expands_titles_and_removes_only_display_markers(display_materials, raw, expected):
    """展示层补齐已知法名、避免重复书名号，不为未知编号补造法律身份或改动普通型号。"""
    assert service_module.clean_display_answer(raw, display_materials) == expected
    assert [m.material_id for m in display_materials] == ["M1", "M2"]


def test_service_cleans_final_answer_but_keeps_raw_ids_and_citation_cards(display_materials):
    """整轮服务必须先通过原始引用校验，再清理展示；审计原文与两张卡片编号保持完整。"""
    raw = reading(quote=display_materials[0].text,
                  extra=f'材料[M2]原文：“{display_materials[1].text}”').replace('USED_IDS: M1','USED_IDS: M1 M2')

    class SearchWithMaterials:
        rows = [m.__dict__ for m in display_materials]

        async def search(self, query, top_k=4):
            return self.rows, "lexical"

    class Model:
        async def chat(self, messages):
            self.messages = messages
            return raw

    async def run():
        model = Model()
        service = ChatService(SearchWithMaterials(), model)
        sid = service.create()
        rid = service.start(sid, "毕业证如何申诉")
        await service.sessions[sid].jobs[rid]["task"]
        job = service.result(sid, rid)
        assert job["status"] == "done", job
        result = job["result"]
        assert not result["blocked"] and not result["raw_output_blocked"]
        assert "《中华人民共和国教育法》" in result["answer"] and display_materials[0].text in result['answer']
        assert all(marker not in result["answer"] for marker in ("[M1]", "[M2]", "USED_IDS"))
        assert result["raw_answer"] == raw
        assert [card["material_id"] for card in result["citations"]] == ["M1", "M2"]
        assert [card["text"] for card in result["citations"]] == [m.text for m in display_materials]
        assert '材料[M1]原文' in model.messages[0]['content']

    asyncio.run(run())


def test_inline_declaration_is_audited_and_cleaned_end_to_end(display_materials):
    """模型不换行也须正确审计三张卡片；展示去声明但保留句号，审计原文不变。"""
    materials = display_materials + [replace(display_materials[1], material_id="M3")]
    raw = reading(quote=materials[0].text,extra='\n'.join(f'材料[{m.material_id}]原文：“{m.text}”' for m in materials[1:]))
    raw = raw.replace('USED_IDS: M1','USED_IDS: M1 M2 M3').replace('\nUSED_IDS:','USED_IDS:')
    assert parse_used_ids(raw) == ["M1", "M2", "M3"]
    checked = postprocess(raw, [m.__dict__ for m in materials])
    assert not checked["blocking"]
    assert not any(issue["type"] == "MISSING_USED_IDS" for issue in checked["suspected_issues"])
    assert service_module.clean_display_answer(raw, materials).endswith('待核实事项并不穷尽。')

    class SearchWithMaterials:
        rows = [m.__dict__ for m in materials]

        async def search(self, query, top_k=4):
            return self.rows[:top_k], "lexical"

    class Model:
        async def chat(self, messages):
            return raw

    async def run():
        service = ChatService(SearchWithMaterials(), Model())
        sid = service.create()
        rid = service.start(sid, "毕业证如何申诉")
        await service.sessions[sid].jobs[rid]["task"]
        job = service.result(sid, rid)
        assert job["status"] == "done", job
        result = job["result"]
        assert not result["blocked"] and result["raw_answer"] == raw
        assert "USED_IDS" not in result["answer"] and result["answer"].endswith('待核实事项并不穷尽。')
        assert not any(issue["type"] == "MISSING_USED_IDS" for issue in result["warnings"])
        assert [card["material_id"] for card in result["citations"]] == ["M1", "M2", "M3"]
        assert service.sessions[sid].history[-1]["content"] == result["answer"]

    asyncio.run(run())


@pytest.mark.parametrize("raw, issue", [
    ("依据[M99]第四十三条处理。\nUSED_IDS: M99", "UNKNOWN_ID"),
    ("依据[M1]第四十三条，须在两日内申诉[M1]。\nUSED_IDS: M1", "DEADLINE_VALUE_NOT_IN_CITED_TEXT"),
])
def test_display_cleanup_cannot_bypass_raw_citation_or_deadline_validation(display_materials, raw, issue):
    """可被展示清理隐藏的非法编号或虚构期限，仍必须在原始回答上被拦截。"""
    class SearchWithMaterials:
        rows = [m.__dict__ for m in display_materials]

        async def search(self, query, top_k=4):
            return self.rows, "lexical"

    class Model:
        async def chat(self, messages):
            return raw

    async def run():
        service = ChatService(SearchWithMaterials(), Model())
        sid = service.create()
        rid = service.start(sid, "毕业证如何申诉")
        await service.sessions[sid].jobs[rid]["task"]
        job = service.result(sid, rid)
        assert job["status"] == "done", job
        result = job["result"]
        assert result["blocked"] and result["raw_output_blocked"]
        assert result["raw_answer"] == raw and result["citations"] == []
        assert any(w["type"] == issue for w in result["warnings"])
        assert service.sessions[sid].history == []

    asyncio.run(run())


def test_external_model_endpoint_rejected():
    """测试安全限制：非本地 HTTP 环回地址（127.0.0.1/localhost）时必须抛出异常拒绝连接"""
    for url in ["https://api.example.com", "http://192.168.1.2:11434", "http://localhost.evil:11434"]:
        with pytest.raises(ValueError):
            OllamaClient(url, "qwen3:4b", "qwen3-embedding:0.6b")


def test_cancel_does_not_commit_partial_answer():
    """测试取消任务逻辑：用户点击停止分析后，任务状态标为 cancelled，且不会将未完成回答写入历史"""
    async def run():
        model = FakeModel()
        s = ChatService(Search(), model)
        sid = s.create()
        rid = s.start(sid, "工伤怎么办")
        await asyncio.sleep(0.01)
        # 用户发起取消
        await s.cancel(sid, rid)
        assert s.result(sid, rid)["status"] == "cancelled"
        assert s.sessions[sid].history == []
        assert model.cancelled

    asyncio.run(run())


def test_reset_invalidates_old_session_and_late_write():
    """测试会话重置与晚写防御：重置旧会话后，旧任务即使随后完成，也绝不能污染新会话"""
    async def run():
        m = FakeModel()
        s = ChatService(Search(), m)
        sid = s.create()
        s.start(sid, "劳动咨询")
        await asyncio.sleep(0.01)
        # 重置会话
        new = await s.reset(sid)
        m.gate.set()  # 放行旧任务
        await asyncio.sleep(0.01)
        assert new != sid and sid not in s.sessions
        assert s.sessions[new].history == []
        with pytest.raises(SessionError):
            s.start(sid, "继续")

    asyncio.run(run())

    asyncio.run(run())


def test_history_is_only_within_session_and_unknown_ids_blocked():
    async def run():
        m = FakeModel()
        m.gate.set()
        s = ChatService(Search(), m)
        sid = s.create()
        other = s.create()
        rid = s.start(sid, "第一轮事实")
        await asyncio.sleep(0.02)
        assert s.result(sid, rid)["status"] == "done"
        assert s.sessions[other].history == []
        s.start(sid, "第二轮补充")
        await asyncio.sleep(0.02)
        assert "第一轮事实" in str(m.messages)
        with pytest.raises(SessionError):
            s.result(other, rid)
    asyncio.run(run())


def test_expired_session_cannot_be_revived_by_request():
    async def run():
        s = ChatService(Search(), FakeModel(), ttl=0.01)
        sid = s.create()
        await asyncio.sleep(0.02)
        with pytest.raises(SessionError):
            s.start(sid, "hello")
        new = await s.reset(sid)
        assert new in s.sessions
    asyncio.run(run())


def test_timeout_is_finite_and_does_not_save_answer():
    async def run():
        s = ChatService(Search(), FakeModel(), timeout=0.02)
        sid = s.create()
        rid = s.start(sid, "工伤")
        await asyncio.sleep(0.06)
        result = s.result(sid, rid)
        assert result["status"] == "error" and "超时" in result["error"]
        assert s.sessions[sid].history == []
    asyncio.run(run())


def test_mixed_labor_issue_still_retrieves_and_old_domain_does_not_poison():
    class Tracking(Search):
        def __init__(self): self.queries = []
        async def search(self, query, top_k=5):
            self.queries.append(query)
            return [], "lexical"
    async def run():
        r, m = Tracking(), FakeModel()
        m.gate.set()
        s = ChatService(r, m)
        sid = s.create()
        s.sessions[sid].history = [{"role":"user","content":"诈骗怎么判刑"}, {"role":"assistant","content":"资料不足"}]
        rid = s.start(sid, "老板说我诈骗所以不给工资")
        await asyncio.sleep(.03)
        assert r.queries
        assert s.result(sid, rid)["status"] == "done"
    asyncio.run(run())


def _prompt_size(messages):
    """与 service.size() 相同的记账口径：UTF-8 内容字节 + 每条消息 32 字节。"""
    return sum(len(m["content"].encode("utf-8")) + 32 for m in messages)


class BudgetSearch:
    """带材料的检索桩：rows 提供领域覆盖，search 返回固定条文并记录 query。"""

    rows = [
        dict(id="m1", kind="LAW_ARTICLE", title="中华人民共和国劳动合同法", locator="第八十五条",
             version_id="2012-12-28", source_url="https://www.gov.cn/flfg/2012-12/28/content_1.htm",
             text="第八十五条 用人单位有下列情形之一的，由劳动行政部门责令限期支付劳动报酬、加班费或者经济补偿；"
                  "劳动报酬低于当地最低工资标准的，应当支付其差额部分；逾期不支付的，责令用人单位按应付金额百分之五十以上"
                  "百分之一百以下的标准向劳动者加付赔偿金：（一）未按照劳动合同的约定或者国家规定及时足额支付劳动者劳动报酬的；"
                  "（二）低于当地最低工资标准支付劳动者工资的；（三）安排加班不支付加班费的；"
                  "（四）解除或者终止劳动合同，未依照本法规定给予经济补偿的。",
             verified_scope="与已保存原文快照一致", content_scope="FULL_TEXT", field="LABOR"),
        dict(id="m2", kind="LAW_ARTICLE", title="中华人民共和国劳动合同法", locator="第四十七条",
             version_id="2012-12-28", source_url="https://www.gov.cn/flfg/2012-12/28/content_1.htm",
             text="第四十七条 经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。"
                  "六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。",
             verified_scope="与已保存原文快照一致", content_scope="FULL_TEXT", field="LABOR"),
    ]

    def __init__(self):
        self.queries = []

    async def search(self, query, top_k=4):
        self.queries.append({"query": query, "top_k": top_k})
        return [dict(row) for row in self.rows[:top_k]], "lexical"


def test_oversized_history_degrades_without_raising_and_keeps_material():
    """超长历史不再报错退出：降级到预算内，材料保底仍在，总字节不超过 7000。"""
    async def run():
        m = FakeModel()
        m.gate.set()
        s = ChatService(BudgetSearch(), m)
        sid = s.create()
        s.sessions[sid].history = [{"role": "user", "content": "长" * 4000}]
        rid = s.start(sid, "继续")
        await s.sessions[sid].jobs[rid]["task"]
        job = s.result(sid, rid)
        assert job["status"] == "done", job
        assert len(job["result"]["retrieved"]) >= 1
        assert _prompt_size(m.messages) <= 7000
    asyncio.run(run())


def test_single_turn_query_and_messages_are_byte_identical():
    """钉扎：history 为空时，检索问句与送模 messages 必须与改造前快照逐字节一致。

    快照由改造前的 service.py 生成（tests_local/fixtures/single_turn_messages.json）。
    任何往单轮输入里加固定块、改缺口文案或改材料取法的改动都会在这里失败。
    """
    fixture = json.loads(
        (pathlib.Path(__file__).parent / "fixtures" / "single_turn_messages.json").read_text(encoding="utf-8"))

    async def run():
        for case in fixture["cases"]:
            retriever = BudgetSearch() if case["name"] == "two_materials" else Search()
            retriever.rows = BudgetSearch.rows if case["name"] == "two_materials" else []
            model = FakeModel()
            model.gate.set()
            service = ChatService(retriever, model, official_catalog=[])
            sid = service.create()
            rid = service.start(sid, case["question"])
            await service.sessions[sid].jobs[rid]["task"]
            job = service.result(sid, rid)
            assert job["status"] == "done", job
            if hasattr(retriever, "queries"):
                assert retriever.queries[0]["query"] == case["retrieval_query"]
            assert model.messages == case["messages"], f"{case['name']} 送模消息与快照不一致"
            assert _prompt_size(model.messages) == case["total_bytes"]

    asyncio.run(run())


def test_twelve_rounds_keep_material_and_budget_without_crashing():
    """连续 12 轮：不报错、每轮至少一条完整材料、总字节不超过 7000，且历史按全文保存。"""
    async def run():
        model = FakeModel()
        model.gate.set()
        service = ChatService(BudgetSearch(), model, official_catalog=[])
        sid = service.create()
        questions = [f"第{i}轮：公司辞退我不给工资，还想让我自己写离职申请" for i in range(1, 13)]
        seen = []
        for index, question in enumerate(questions, 1):
            rid = service.start(sid, question)
            await service.sessions[sid].jobs[rid]["task"]
            job = service.result(sid, rid)
            assert job["status"] == "done", (index, job)
            assert len(job["result"]["retrieved"]) >= 1, index
            assert _prompt_size(model.messages) <= 7000, index
            assert question in str(model.messages), index
            seen.append(len(job["result"]["retrieved"]))
        history = service.sessions[sid].history
        assert len(history) == 24
        # 历史保留回答全文，压缩只发生在装配时；否则引用审计与“重新开始”前的核对都会失真。
        assert max(len(m["content"]) for m in history if m["role"] == "assistant") > 120

    asyncio.run(run())


def test_overlong_question_is_truncated_for_prompt_but_keeps_material():
    """问题本身超长时：截断的只是送模副本，材料仍保底，且显式告知用户已截断。"""
    async def run():
        model = FakeModel()
        model.gate.set()
        retriever = BudgetSearch()
        service = ChatService(retriever, model, official_catalog=[])
        sid = service.create()
        # 选择提示比旧阅读提示短，原 1600 字问题已能装进 7000 字节；
        # 这里加长到仍会触发“问题本身超长”的兜底路径，测试意图不变。
        question = "问" * 2000
        rid = service.start(sid, question)
        await service.sessions[sid].jobs[rid]["task"]
        job = service.result(sid, rid)
        assert job["status"] == "done", job
        assert len(job["result"]["retrieved"]) >= 1
        assert _prompt_size(model.messages) <= 7000
        assert retriever.queries[0]["query"] == question
        assert any("过长" in gap for gap in job["result"]["gaps"])

    asyncio.run(run())


def _reading_system(question):
    """取得真实服务发出的阅读系统消息，同时验证通用回答能正常放行。"""
    async def run():
        model = FakeModel()
        model.gate.set()
        service = ChatService(Search(), model)
        sid = service.create()
        rid = service.start(sid, question)
        await service.sessions[sid].jobs[rid]['task']
        job = service.result(sid, rid)
        assert job['status'] == 'done' and not job['result']['blocked'], job
        assert job['result']['answer_mode'] == 'material_reading'
        assert sum(len(m['content'].encode('utf-8')) + 32 for m in model.messages) <= 7000
        return model.messages[0]['content']
    return asyncio.run(run())


def test_model_system_requires_source_bound_material_reading():
    """阅读许可取代自由处置路径；同样保持四标题与真实消息预算。"""
    system = _reading_system('请根据材料说明还缺哪些客观事实')
    for rule in ('只读本轮材料', '不得判断提问人或对方的责任', '完整短句'):
        assert rule in system
    headings = ['## 追问', '## 处理方向', '## 证据清单', '## 不确定性']
    assert [system.index(h) for h in headings] == sorted(system.index(h) for h in headings)


def test_model_system_does_not_unlock_verdicts_after_user_supplies_facts():
    """事实、权属或条件补充不等同于程序确认法律适用。"""
    system = _reading_system('权属和技术经过都说清了，现在直接确定责任吧')
    assert '用户说条件已补齐也不解锁个案结论' in system
    assert '否定结论和如果句同样禁止' in system
    assert '待核实事项并不穷尽' in system


def test_model_system_requires_whole_source_units_and_complete_ids():
    """模型原文必须可逐字核对；编号声明沿用原始审计，不靠展示清理放行。"""
    system = _reading_system('请说明材料写了什么')
    for rule in ('材料[M1]原文', '完整句或段落', '不删否定、条件、但是、除外',
                 '正文实际出现的全部编号', '没有使用材料写USED_IDS: 无', '650个汉字以内'):
        assert rule in system
    assert len(system.encode('utf-8')) <= 3376


def test_model_system_permits_generic_steps_without_person_specific_routes():
    """通用保存/核实步骤保留，不再要求模型在有限材料中替用户选择诉讼路径。"""
    system = _reading_system('路上发生了纠纷，想先了解材料')
    assert '1. 保存已有文件、凭证和沟通记录。' in system
    assert '2. 到官方文本核对版本、主体、条件及例外。' in system
    assert '刑事犯罪指控不得以民事反诉处理' in system
    assert '不得判断' in system and '具体诉讼请求' in system


def test_longer_system_prompt_still_fits_input_budget_by_whole_materials():
    """新增提示词也计入输入预算；大候选只能整条移除，不能截短原文后送模型。"""
    class LargeMaterials:
        rows = [dict(
            id=str(i), kind="LAW_ARTICLE", title=f"测试规范{i}", locator="第一条",
            text="测试原文" * 140, version_id="测试版本", source_url="https://www.gov.cn/",
            verified_scope="测试替身", content_scope="ARTICLES_ONLY",
        ) for i in range(4)]

        async def search(self, query, top_k=4):
            return self.rows[:top_k], "lexical"

    async def run():
        model = FakeModel()
        model.gate.set()
        service = ChatService(LargeMaterials(), model)
        sid = service.create()
        rid = service.start(sid, "请结合资料说明可行的处理方向")
        await service.sessions[sid].jobs[rid]["task"]
        job = service.result(sid, rid)
        assert job["status"] == "done", job
        assert sum(len(message["content"].encode("utf-8")) + 32 for message in model.messages) <= 7000
        supplied = job["result"]["retrieved"]
        assert 0 < len(supplied) < 4
        assert all(row["text"] == "测试原文" * 140 for row in supplied)
        assert any("部分候选材料因上下文预算未送入模型" in gap for gap in job["result"]["gaps"])

    asyncio.run(run())


def test_model_system_permits_insufficient_material_opt_out():
    """没有直接对应的材料时可以只给通用核实步骤，不强迫引用。"""
    system = _reading_system('材料与所问情况不对应时怎么办')
    assert '本轮材料均不直接对应时用' in system
    assert '本轮材料不足以直接说明所问情况' in system
    assert 'USED_IDS: 无' in system


@pytest.mark.parametrize("question, region, keyword", [
    ("我想申请上海市徐汇区2026年人才租房补贴，请按该区当年的实施细则告诉我收入门槛和补贴金额。",
     "上海市徐汇区", "徐汇区"),
    ("孩子在深圳市南山区读幼儿园，申请2026年学位补贴需要什么条件？请以南山区当年的补贴实施细则为准。",
     "深圳市南山区", "南山区"),
])
def test_uncovered_local_subsidy_degrades_without_unrelated_cards(question, region, keyword):
    """无地方标题时，即使召回反补贴条例且模型虚构金额，也须给出可评分的范围降级。"""
    class SubsidySearch:
        rows = [dict(
            kind="LAW_ARTICLE", title="中华人民共和国反补贴条例", locator="第一条",
            text="为了维护对外贸易秩序和公平竞争，制定本条例。", version_id="测试版本",
            source_url="https://www.gov.cn/", content_scope="ARTICLES_ONLY", verified_scope="测试替身",
        )]

        async def search(self, query, top_k=4):
            return self.rows[:top_k], "lexical"

    class UnsafeModel:
        async def chat(self, messages):
            # 明确检查无关候选在组装模型材料前已经被清除。
            self.messages = messages
            return "补贴为3000元，申请条件是本区户籍。\nUSED_IDS: M1"

    async def run():
        model = UnsafeModel()
        service = ChatService(SubsidySearch(), model)
        sid = service.create()
        rid = service.start(sid, question)
        await service.sessions[sid].jobs[rid]["task"]
        job = service.result(sid, rid)
        assert job["status"] == "done", job
        result = job["result"]
        assert result["scope_degraded"] and not result["blocked"]
        assert region in result["answer"]
        assert judge_degrade(result["answer"], result["blocked"], [keyword])
        assert "申请条件" in result["answer"] and "金额" in result["answer"]
        assert result["citations"] == [] and result["retrieved"] == []
        assert "反补贴条例" not in str(model.messages)
        assert "3000元" not in result["answer"] and "3000元" in result["raw_answer"]
        assert result["raw_output_blocked"]

    asyncio.run(run())


@pytest.mark.parametrize("question, titles, expected", [
    ("请按上海市徐汇区人才租房补贴实施细则说明金额", [], "上海市徐汇区"),
    ("请按上海市徐汇区人才租房补贴实施细则说明金额", ["上海市人才租房补贴规则"], "上海市徐汇区"),
    ("请按上海市徐汇区人才租房补贴实施细则说明金额", ["上海市徐汇区人才租房补贴实施细则"], None),
    ("请按深圳市南山区学位补贴实施细则说明申请条件", ["深圳市福田区学位补贴实施细则"], "深圳市南山区"),
    ("请按杭州市人才租房补贴实施细则说明申请条件", [], "杭州市"),
    ("请按安吉县租房补贴实施细则说明申请条件", [], "安吉县"),
    ("请按徐汇区租房补贴实施细则说明金额", [], "徐汇区"),
    ("请按南京市鼓楼区租房补贴实施细则说明条件", ["福州市鼓楼区租房补贴实施细则"], "南京市鼓楼区"),
    ("请按南京市鼓楼区租房补贴实施细则说明条件", ["南京市鼓楼区租房补贴实施细则"], None),
    ("请按浙江省杭州市余杭区租房补贴实施细则说明条件", ["杭州市余杭区租房补贴实施细则"], None),
    ("在上海市申请人才租房补贴，条件是什么", ["上海市人才租房补贴实施细则"], None),
    ("我能在上海市申请人才租房补贴吗，条件是什么", ["上海市人才租房补贴实施细则"], None),
    ("不要按深圳市南山区的规则，请按上海市徐汇区人才租房补贴实施细则说明条件",
     ["上海市徐汇区人才租房补贴实施细则"], None),
    ("不要按深圳市南山区的规则而是按上海市徐汇区人才租房补贴实施细则说明条件",
     ["上海市徐汇区人才租房补贴实施细则"], None),
    ("我在上海市徐汇区，请按中国法律说明经济补偿", [], None),
    ("请按中国法律说明上海市徐汇区租房补贴申请条件", [], "上海市徐汇区"),
    ("请解释反补贴条例的全国适用范围", [], None),
    ("上海市的反补贴条例属于全国规则吗", [], None),
    ("我们小区租房补贴的申请条件是什么", [], None),
    ("租房补贴的申请条件是什么", [], None),
])
def test_local_rule_scope_respects_specific_region_and_title_coverage(question, titles, expected):
    """具体区县必须单独覆盖；同市其他区、全国条例和普通小区都不能冒充地方细则。"""
    rows = [{"title": title} for title in titles]
    assert uncovered_foreign_region(question, rows) == expected


def test_scope_degrade_requires_shared_validator_acceptance(monkeypatch):
    """共享降级判定拒绝时不得发布回复或写入历史，防止绕开最终校验。"""
    monkeypatch.setattr("local_rag.service.judge_degrade", lambda *args: False)

    async def run():
        model = FakeModel()
        model.gate.set()
        service = ChatService(Search(), model)
        sid = service.create()
        rid = service.start(sid, "请按上海市徐汇区人才租房补贴实施细则说明申请条件")
        await service.sessions[sid].jobs[rid]["task"]
        job = service.result(sid, rid)
        assert job["status"] == "error" and "未通过降级校验" in job["error"]
        assert "result" not in job and service.sessions[sid].history == []

    asyncio.run(run())
