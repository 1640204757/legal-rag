"""Web真实缺陷的确定性回归：声明容错、自愈边界与刑民路径/场景护栏。"""
import asyncio

import pytest

from local_rag.service import ChatService, clean_display_answer, answer_scenario_issues
from prototype.rag.input_builder import Material
from prototype.rag.citation_check import parse_used_ids, find_citations, check_ids
from prototype.rag.postprocess import postprocess


@pytest.fixture
def evidence():
    """返回两条同法不同条号材料，专门检查仅写法名时不能随意补编号。"""
    return [Material(material_id=f"M{i}", kind="LAW_ARTICLE", title="中华人民共和国刑法", locator=article,
                     text=article+" "+text, version_id="测试版本", source_url="https://www.gov.cn/test",
                     verified_scope="仅测试") for i, (article, text) in enumerate([
                         ("第二百三十七条", "强制猥亵他人"), ("第二百四十三条", "捏造事实诬告陷害他人")], 1)]


@pytest.mark.parametrize("tail", ["USED_IDS:M1 M2", "USED_IDS M1,M2", "USED_IDS： M 1、M 2",
                                  "used_ids:M1 M2", "USED_IDS: 1, 2", "USED_IDSM1M2", "USED_IDS:M1M2"])
def test_declaration_variants_keep_punctuation_and_hide_all_internal_text(evidence, tail):
    raw = "结论[M1][M2]。"+tail
    assert parse_used_ids(raw) == ["M1", "M2"]
    assert find_citations(raw) == ["M1", "M2"]
    assert clean_display_answer(raw, evidence) == "结论。"


def test_bare_material_reference_is_displayed_as_law_name(evidence):
    assert clean_display_answer('依据 M1 核对。USED_IDS:M1', evidence) == '依据 《中华人民共和国刑法》 核对。'


@pytest.mark.parametrize('variable', ['USED_IDS_DEBUG', 'MY_USED_IDS', 'USED_IDS_VERSION'])
def test_identifier_names_are_not_declarations_or_deleted(evidence, variable):
    raw = variable+' = M99; 这是日志变量。'
    assert parse_used_ids(raw) is None
    assert variable in clean_display_answer(raw, evidence)


def test_declaration_without_any_material_cannot_pass_low_level_check():
    assert check_ids('USED_IDS M99', [])['verdict'] == 'FAIL'


@pytest.mark.parametrize("tail", ["", "\nUSED_IDS: 无", "。USED_IDS M1"])
def test_legal_body_ids_heal_declaration_without_modifying_raw(evidence, tail):
    raw = "请核对[M1]并固定证据[M2]"+tail
    checked = postprocess(raw, [m.__dict__ for m in evidence])
    assert not checked["blocking"]
    assert checked["raw_answer"] == raw and checked["answer"] == raw and not checked["modified"]
    assert checked["used_ids"] == ["M1", "M2"]
    assert checked["declaration_repair"]["repaired"]
    assert not {i["type"] for i in checked["suspected_issues"]} & {"MISSING_USED_IDS", "ID_CITED_BUT_NOT_DECLARED"}
    assert [c["material_id"] for c in checked["citation_cards"]] == ["M1", "M2"]


def test_unambiguous_named_article_heals_but_unknown_and_ambiguous_do_not(evidence):
    raw = "依据《刑法》第二百三十七条核对。"
    checked = postprocess(raw, [m.__dict__ for m in evidence])
    assert not checked["blocking"] and checked["used_ids"] == ["M1"]
    for raw in ("依据《刑法》第999条处理。", "正文[M1]。USED_IDS M99", "正文[M99]。USED_IDS: M1"):
        assert postprocess(raw, [m.__dict__ for m in evidence])["blocking"]
    ambiguous = postprocess("依据《刑法》处理。", [m.__dict__ for m in evidence])
    assert not ambiguous["declaration_repair"]["repaired"]


@pytest.mark.parametrize("answer", ["可以提起民事反诉追究诬告。", "建议反诉她们强奸。",
                                    "不要犹豫直接提起民事反诉追究诬告。", "不能只报警还应提起民事反诉。",
                                    "向单位法务反映。", "主张单位管理责任并找学生申诉。"])
def test_public_criminal_event_rejects_wrong_routes_and_templates(answer):
    assert answer_scenario_issues("回家路上被强制猥亵，她们反咬告我强奸怎么办", answer)


@pytest.mark.parametrize("answer", ["不能提起民事反诉，应向公安机关报案并如实陈述。",
                                    "先脱离危险就医验伤，保留监控线索，依法委托辩护人。",
                                    "如有材料支持，可依法另行主张人身损害赔偿。"])
def test_criminal_route_guard_preserves_negative_reminders_and_civil_compensation(answer):
    assert not answer_scenario_issues("回家路上被猥亵又被诬告", answer)
    assert not answer_scenario_issues("学校处分不服", "向学生申诉委员会申诉。")


def test_independent_civil_case_is_separated_from_criminal_accusation():
    question = '对方已以借款起诉我，又到公安报案说我诈骗，两个事怎么处理'
    assert not answer_scenario_issues(question, '在该借款民事案件中，如有独立欠款请求可提出反诉；诈骗指控向公安如实陈述。')
    assert answer_scenario_issues(question, '针对诈骗刑事指控提起民事反诉。')


def test_service_repairs_ids_and_blocks_wrong_route_before_display(evidence):
    from tests_local.test_answer_permission import reading
    class Search:
        rows = [m.__dict__ for m in evidence]
        async def search(self, query, top_k=4):
            return self.rows, "lexical"
    class Model:
        # 使用当前阅读格式验证“合法原文仍可自动补齐声明”，不再把自由个案建议当正例。
        raw = reading(quote=evidence[0].text,
                      extra=f'材料[M2]原文：“{evidence[1].text}”').replace('USED_IDS: M1', '')
        async def chat(self, messages):
            self.messages = messages
            return self.raw
    async def run():
        model = Model()
        service = ChatService(Search(), model, official_catalog=[])
        sid = service.create()
        for bad in (False, True):
            model.raw = "可以提起民事反诉[M1]。USED_IDS: M1" if bad else model.raw
            rid = service.start(sid, "回家路上被猥亵后反咬告我强奸")
            await service.sessions[sid].jobs[rid]["task"]
            result = service.result(sid, rid)["result"]
            assert result["raw_answer"] == model.raw
            assert result["blocked"] is bad
            if bad:
                assert result["citations"] == [] and "民事反诉" not in result["answer"]
            else:
                assert result["declaration_repair"]["repaired"]
                assert len(result["citations"]) == 2 and "USED_IDS" not in result["answer"]
        assert "刑事犯罪指控不得以民事反诉处理" in model.messages[0]["content"]
    asyncio.run(run())


def test_source_reading_supplies_only_verified_deadline_evidence():
    from tests_local.test_answer_permission import reading
    material = Material(material_id='M1', kind='LAW_ARTICLE', title='中华人民共和国劳动合同法', locator='第三十七条',
                        text='第三十七条 劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。',
                        version_id='测试', source_url='https://www.gov.cn/test', verified_scope='测试')
    class Search:
        rows = [material.__dict__]
        async def search(self, query, top_k=4): return self.rows, 'lexical'
    class Model:
        raw = reading(quote=material.text)
        async def chat(self, messages): return self.raw
    async def run():
        model = Model()
        service = ChatService(Search(), model, official_catalog=[])
        for duration, blocked in [('三十日', False), ('两日', True)]:
            model.raw = reading(quote=material.text.replace('三十日',duration))
            sid = service.create(); rid = service.start(sid, '辞职怎么通知公司')
            await service.sessions[sid].jobs[rid]['task']
            assert service.result(sid,rid)['result']['blocked'] is blocked
    asyncio.run(run())


def test_new_guard_prompt_does_not_crowd_out_material_budget(evidence):
    """保留新刑民/主体规则同时限制系统字节，避免给四条材料增加无法承受的固定开销。"""
    class Search:
        rows = [m.__dict__ for m in evidence]
        async def search(self, query, top_k=4): return self.rows, 'lexical'
    class Model:
        async def chat(self, messages):
            self.messages = messages
            return 'USED_IDS: 无'
    async def run():
        model = Model(); service = ChatService(Search(), model, official_catalog=[])
        sid=service.create(); rid=service.start(sid, '回家路上被猥亵后诬告怎么办')
        await service.sessions[sid].jobs[rid]['task']
        system = model.messages[0]['content']
        assert '刑事犯罪指控不得以民事反诉处理' in system
        assert len(system.encode('utf-8')) <= 3376
    asyncio.run(run())
