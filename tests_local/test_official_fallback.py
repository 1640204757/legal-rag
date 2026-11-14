"""官方目录双流程回归：离线缺口/版本盘点与在线目录定位、受控补充相互独立。

纯函数用合成元数据与真实目录验证；服务用异步抓取替身，不访问网络或修改法规快照。
主阅读入口为目录盘点、真实主题匹配及三种服务兜底测试。
"""
import asyncio
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import local_rag.expansion_batches as expansion
import local_rag.retrieval as retrieval
from local_rag.official_capture import public_url
from local_rag.service import ChatService


TITLE = "最高人民法院关于人民法院办理财产保全案件若干问题的规定"
ENTRY = dict(title=TITLE, bbbs="a" * 32, gbrq="2020-12-29", sxrq="2021-01-01", sxx=3, flxz="司法解释")
QUESTION = "申请财产保全时，财产保全司法解释对申请材料有什么规定"
SCOPE = "官方白名单实时检索补充；已核验官方来源与条号，未独立复核个案适用"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("indexed, expected", [
    ([], "MISSING_FROM_LOCAL"),
    ([dict(title=TITLE, publication_date="2019-01-01", effective_date="2021-01-01")], "VERSION_UPDATED_ON_OFFICIAL"),
    ([dict(title=TITLE, publication_date="2020-12-29", effective_date="2020-01-01")], "VERSION_UPDATED_ON_OFFICIAL"),
    ([dict(title=TITLE, version="2020-12-29", effective_date="2021-01-01")], None),
    ([dict(title=TITLE, publication_date="2022-01-01", effective_date="2022-01-01")], None),
    ([dict(title=TITLE)], None),
])
def test_catalog_updates_compare_missing_and_known_dates_without_mutation(indexed, expected):
    """只在同名缺失或可比较日期确实变新时报告；缺日期不冒充已经证明新版。"""
    entries = [deepcopy(ENTRY)]
    before = deepcopy((entries, indexed))
    result = expansion.find_catalog_updates(entries, indexed)
    assert (entries, indexed) == before
    if expected is None:
        assert result == []
    else:
        assert len(result) == 1
        assert result[0]["update_type"] == expected
        assert result[0]["official_url"] == "https://flk.npc.gov.cn/detail?id=" + "a" * 32
        assert result[0]["publication_date"] == "2020-12-29"
        assert result[0]["effective_date"] == "2021-01-01"
        assert result[0]["bbbs"] == "a" * 32


def test_catalog_update_normalizes_title_and_uses_latest_indexed_dates():
    """空白/书名号不能制造假缺口，重复条文元数据不能盖住已入库新版本。"""
    indexed = [dict(title="《 " + TITLE + " 》", version="2019-01-01"),
               dict(title=TITLE, publication_date="2020-12-29", effective_date="2021-01-01")]
    assert expansion.find_catalog_updates([ENTRY], indexed) == []


@pytest.mark.parametrize("changes", [dict(sxx=1), dict(sxx=2), dict(sxx=None), dict(status="已废止"),
                                     dict(title=TITLE + "（征求意见稿）"), dict(status="失效")])
def test_catalog_update_filters_inactive_or_draft_entries(changes):
    """官方状态字段优先，标题或其他状态明确失效/草稿也不得进入候选。"""
    assert expansion.find_catalog_updates([dict(ENTRY, **changes)], []) == []


@pytest.mark.parametrize("url", ["http://flk.npc.gov.cn/detail?id=x", "https://example.com/law", "https://court.gov.cn.evil.test/x",
                                "https://evilgov.cn/x", "https://user:pass@www.court.gov.cn/x", "https://www.court.gov.cn:9443/x",
                                "https://www.co\nurt.gov.cn/x", "https://@www.court.gov.cn/x", "https://:@www.court.gov.cn/x"])
def test_catalog_update_rejects_unsafe_explicit_urls_even_with_npc_id(url):
    """存在恶意显式URL时不能退回bbbs构造安全链接来掩盖来源异常。"""
    assert expansion.find_catalog_updates([dict(ENTRY, url=url)], []) == []


@pytest.mark.parametrize("url", ["https://xzfg.moj.gov.cn/front/law/detail?LawID=1", "https://www.court.gov.cn/zixun/1.html",
                                "https://example.gov.cn/law"])
def test_administrative_catalog_and_opt_in_official_domains(url):
    """行政目录不依赖bbbs，官方后缀须显式允许；不扩展旧采集器默认主机表。"""
    entry = dict(title="测试行政法规", url=url, publication_date="2024年1月2日", effective_date="2024-02-01")
    result = expansion.find_catalog_updates([entry], [])
    assert result[0]["official_url"] == url and result[0]["publication_date"] == "2024-01-02"
    assert public_url(url, allow_government=True) == url
    if urlsplit(url).hostname == "example.gov.cn":
        with pytest.raises(ValueError):
            public_url(url)


@pytest.fixture(scope="module")
def judicial_catalog_and_indexed_titles():
    """只读真实目录与核验语料，不计算向量、不写索引。"""
    catalog = json.loads((ROOT / "data/catalogs/official-20261020/npc/judicial/catalog.json").read_text(encoding="utf-8"))["entries"]
    rows, _ = retrieval.load_corpus()
    return catalog, {row["title"] for row in rows}


@pytest.mark.parametrize("question, expected", [
    ("建设工程施工合同司法解释如何处理工程款", "最高人民法院关于审理建设工程施工合同纠纷案件适用法律问题的解释（一）"),
    (QUESTION, TITLE),
    ("民事诉讼证据规定对提供证据有什么要求", "最高人民法院关于民事诉讼证据的若干规定"),
    ("网络消费司法解释怎么规定网络消费退款", "最高人民法院关于审理网络消费纠纷案件适用法律若干问题的规定（一）"),
    ("人脸识别技术获取个人信息的司法解释", "最高人民法院关于审理使用人脸识别技术处理个人信息相关民事案件适用法律若干问题的规定"),
    ("生态环境侵权民事诉讼证据有什么专门规定", "最高人民法院关于生态环境侵权民事诉讼证据的若干规定"),
])
def test_real_official_catalog_matches_missing_special_topics(judicial_catalog_and_indexed_titles, question, expected):
    """真实目录精确定位缺失主题；建设工程解释已入库时先验证排除，再模拟移除该法名的缺口。"""
    catalog, indexed = judicial_catalog_and_indexed_titles
    if expected in indexed:
        existing = retrieval.match_official_whitelist_catalog(question, catalog, indexed, top_k=2)
        assert all(entry['title'] != expected for entry in existing)
    matches = retrieval.match_official_whitelist_catalog(question, catalog, indexed - {expected}, top_k=2)
    assert matches and matches[0]["title"] == expected, matches
    assert len(matches) <= 2
    assert all(urlsplit(entry["official_url"]).hostname == "flk.npc.gov.cn" for entry in matches)


@pytest.mark.parametrize("question", ["法律规定怎么办", "证据", "我想提起民事诉讼", "如何维护自己的权益"])
def test_catalog_matcher_does_not_use_generic_legal_fragments(question):
    """泛法律词、仅两字主题及泛诉讼名称不足以证明命中缺失专项解释。"""
    assert retrieval.match_official_whitelist_catalog(question, [ENTRY], set()) == []


def test_catalog_matcher_excludes_indexed_inactive_unsafe_and_respects_top_k():
    """已入库/失效/恶意源不能重复推荐，零Top-K明确返回空。"""
    assert retrieval.match_official_whitelist_catalog(QUESTION, [ENTRY], {"《 " + TITLE + " 》"}) == []
    assert retrieval.match_official_whitelist_catalog(QUESTION, [dict(ENTRY, sxx=2), dict(ENTRY, url="https://evil.test/law")], set()) == []
    assert retrieval.match_official_whitelist_catalog(QUESTION, [ENTRY], set(), top_k=0) == []


def _slice(**changes):
    """返回完整抓取替身切片；无网络副作用，改动字段用于拒绝边界测试。"""
    row = dict(kind="LAW_ARTICLE", title=TITLE, locator="第一条", text="第一条 当事人申请财产保全，应当提交相关材料。",
               version_id="2020-12-29", source_url="https://www.court.gov.cn/zixun/test.html", content_scope="ARTICLES_ONLY",
               verified_scope="抓取替身标记")
    return dict(row, **changes)


class LocalSearch:
    """内存检索替身：记录已加载法名，按给定候选返回，不联网或写磁盘。"""
    def __init__(self, hits=None, mode="lexical", indexed=None):
        self.hits = hits or []
        self.mode = mode
        self.rows = self.hits if indexed is None else indexed

    async def search(self, query, top_k=4):
        return self.hits[:top_k], self.mode


class Model:
    """记录实际发模消息，仅在送入材料时引用M1；隔离真实模型响应质量。"""
    def __init__(self, cite=False):
        self.cite = cite
        self.messages = []

    async def chat(self, messages):
        self.messages = messages
        from tests_local.test_answer_permission import reading
        if self.cite:
            return reading(quote=_slice()['text'])
        return ("## 追问\n请说明已有的文件、通知或沟通记录。\n## 处理方向\n本轮材料不足以直接说明所问情况。"
                "\n1. 保存已有文件、凭证和沟通记录。\n2. 到官方文本核对版本、主体、条件及例外。"
                "\n## 证据清单\n保存记录。\n## 不确定性\n仍需核实。\nUSED_IDS: 无")


async def _answer(service, question=QUESTION):
    """执行并等待单轮真实服务，返回结果和会话；只改本测试服务的内存状态。"""
    sid = service.create()
    rid = service.start(sid, question)
    await service.sessions[sid].jobs[rid]["task"]
    job = service.result(sid, rid)
    assert job["status"] == "done", job
    return job["result"], sid


@pytest.mark.parametrize("question, indexed", [("工资被拖欠怎么办", []), (QUESTION, [dict(title=TITLE)])])
def test_high_confidence_local_hit_without_missing_topic_does_not_fetch(question, indexed):
    """高分且没有缺失专项主题时不能联网，也不能输出虚假的兜底来源。"""
    async def run():
        calls = []
        async def fetcher(entry, query):
            calls.append(entry)
            return _slice()
        local = _slice(title="中华人民共和国劳动合同法", retrieval_score=100.0)
        model = Model()
        service = ChatService(LocalSearch([local], indexed=indexed), model, official_catalog=[ENTRY], official_fetcher=fetcher)
        result, _ = await _answer(service, question)
        assert calls == [] and result["official_fallback"] == []
    asyncio.run(run())


@pytest.mark.parametrize("mode, score, trigger", [("lexical", 10.0, "LOW_LOCAL_CONFIDENCE"),
                                                  ("hybrid", 0.02, "LOW_LOCAL_CONFIDENCE"),
                                                  ("lexical", 100.0, "MISSING_SPECIAL_TOPIC")])
def test_missing_special_topic_can_supply_verified_official_material(mode, score, trigger):
    """弱本地候选或明确缺失专项主题时补入官方条文；所有材料仍由真实组装和引用检查处理。"""
    async def run():
        calls = []
        async def fetcher(entry, query):
            calls.append((entry, query))
            return _slice()
        # 无关候选超过字符预算时不可截短，官方短切片可补足本轮实际材料。
        local = _slice(title="其他规范", text="第一条 " + "无关内容" * 1000, retrieval_score=score)
        model = Model(cite=True)
        service = ChatService(LocalSearch([local], mode), model, official_catalog=[ENTRY], official_fetcher=fetcher)
        result, _ = await _answer(service)
        assert len(calls) == 1 and calls[0][1] == QUESTION
        assert calls[0][0]["official_url"].startswith("https://flk.npc.gov.cn/detail?id=")
        assert calls[0][0]["url"] == calls[0][0]["official_url"]
        assert not result["blocked"] and not result["scope_degraded"]
        assert result["citations"][0]["verified_scope"] == SCOPE
        assert _slice()["text"] in model.messages[-1]["content"]
        fallback = result["official_fallback"][0]
        assert fallback["trigger"] == trigger and fallback["sent_to_model"]
        assert fallback["material_ids"] == ["M1"]
        assert sum(len(m["content"].encode("utf-8")) + 32 for m in model.messages) <= 7000
    asyncio.run(run())


def test_default_offline_mode_reports_catalog_metadata_without_fabricated_articles():
    """不注入抓取器时仅目录定位；标题、日期和官方链接不能伪装成条文引用。"""
    async def run():
        model = Model()
        service = ChatService(LocalSearch(), model, official_catalog=[ENTRY])
        result, _ = await _answer(service)
        fallback = result["official_fallback"][0]
        assert fallback["title"] == TITLE and fallback["publication_date"] == "2020-12-29"
        assert fallback["effective_date"] == "2021-01-01"
        assert fallback["official_url"] == "https://flk.npc.gov.cn/detail?id=" + "a" * 32
        assert fallback["trigger"] == "LOCAL_EMPTY" and not fallback["sent_to_model"]
        assert result["retrieved"] == [] and result["citations"] == []
        assert any(TITLE in gap and "本地库暂未收录" in gap and "未从非官方站点抓取" in gap for gap in result["gaps"])
    asyncio.run(run())


def test_default_catalog_can_locate_real_missing_topic_without_fetcher():
    """默认读取现有国家库目录快照，普通问答流程本身不联网或写入目录。"""
    async def run():
        service = ChatService(LocalSearch(), Model())
        result, _ = await _answer(service)
        assert result["official_fallback"][0]["title"] == TITLE
        assert result["retrieved"] == [] and not result["official_fallback"][0]["sent_to_model"]
    asyncio.run(run())


@pytest.mark.parametrize("changes", [dict(source_url="https://evil.test/law"), dict(source_url="http://www.court.gov.cn/law"),
                                     dict(title="其他规范"), dict(locator="没有条号"), dict(text="第二条 与第一条不一致。"),
                                     dict(final_url="https://evil.test/redirect"), dict(text="第一条"),
                                     dict(text="第一条   "), dict(text="第一条 。")])
def test_fetched_slice_must_pass_source_title_and_article_checks(changes):
    """官方目录链接不为恶意最终来源、错法名或错条号背书，拒绝时只返回安全目录信息。"""
    async def run():
        async def fetcher(entry, query):
            return _slice(**changes)
        service = ChatService(LocalSearch(), Model(), official_catalog=[ENTRY], official_fetcher=fetcher)
        result, _ = await _answer(service)
        assert result["retrieved"] == [] and result["citations"] == []
        assert not result["official_fallback"][0]["sent_to_model"]
        assert "evil.test" not in str(result["official_fallback"])
    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, RuntimeError("不得向用户暴露此抓取内部细节"), TimeoutError("超时细节")])
def test_fetch_failure_keeps_directory_only_and_does_not_fail_local_chat(failure):
    """未得到条文或抓取失败均安全回退目录定位，不泄露异常载荷。"""
    async def run():
        async def fetcher(entry, query):
            if failure:
                raise failure
            return None
        service = ChatService(LocalSearch(), Model(), official_catalog=[ENTRY], official_fetcher=fetcher)
        result, _ = await _answer(service)
        assert result["official_fallback"] and result["retrieved"] == []
        assert "抓取内部细节" not in str(result)
    asyncio.run(run())


@pytest.mark.parametrize("question", ["请按日本法律说明财产保全的申请条件",
                                      "请按上海市徐汇区财产保全补贴实施细则说明申请条件"])
def test_region_hard_degrade_never_calls_official_fetcher(question):
    """境外和缺失地方补贴规则是硬降级，不能用全国官方目录条目绕过。"""
    async def run():
        calls = []
        async def fetcher(entry, query):
            calls.append(entry)
            return _slice()
        service = ChatService(LocalSearch(), Model(), official_catalog=[ENTRY], official_fetcher=fetcher)
        result, _ = await _answer(service, question)
        assert result["scope_degraded"] and calls == [] and result["official_fallback"] == []
        assert result["citations"] == [] and result["retrieved"] == []
    asyncio.run(run())


def test_untrusted_catalog_never_reaches_fetcher():
    """即使问题命中且本地为空，也不得调用非官方目录地址的适配器。"""
    async def run():
        calls = []
        async def fetcher(entry, query):
            calls.append(entry)
            return _slice()
        service = ChatService(LocalSearch(), Model(), official_catalog=[dict(ENTRY, url="https://evil.test/law")], official_fetcher=fetcher)
        result, _ = await _answer(service)
        assert calls == [] and result["official_fallback"] == []
    asyncio.run(run())


def test_oversized_supplement_is_not_truncated_or_claimed_sent_to_model():
    """抓到条文不代表已送模；超过字符预算只能整条排除并保留目录定位记录。"""
    async def run():
        text = "第一条 " + "补充条文" * 1000
        async def fetcher(entry, query):
            return _slice(text=text)
        model = Model()
        service = ChatService(LocalSearch(), model, official_catalog=[ENTRY], official_fetcher=fetcher)
        result, _ = await _answer(service)
        fallback = result["official_fallback"][0]
        assert not fallback["sent_to_model"] and fallback["material_ids"] == []
        assert result["retrieved"] == [] and text not in str(model.messages)
        assert sum(len(m["content"].encode("utf-8")) + 32 for m in model.messages) <= 7000
    asyncio.run(run())


def test_cancelling_official_fetch_does_not_run_model_or_write_history():
    """用户取消须传播到正在等待的抓取适配器，不能被降级异常分支吞掉后继续推理。"""
    async def run():
        started = asyncio.Event()
        cancelled = []
        async def fetcher(entry, query):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
        model = Model()
        service = ChatService(LocalSearch(), model, official_catalog=[ENTRY], official_fetcher=fetcher)
        sid = service.create()
        rid = service.start(sid, QUESTION)
        await started.wait()
        await service.cancel(sid, rid)
        assert cancelled == [True] and model.messages == []
        assert service.result(sid, rid)['status'] == 'cancelled'
        assert service.sessions[sid].history == []
    asyncio.run(run())


def test_future_version_can_be_inventoried_but_not_used_as_current_online_law():
    """维护盘点可提示已公布待施行版本，在线不得把目录未来日期的条文作为现行依据送模。"""
    async def run():
        future = (date.today() + timedelta(days=30)).isoformat()
        entry = dict(ENTRY, sxrq=future)
        assert expansion.find_catalog_updates([entry], [dict(title=TITLE, version='2019-01-01')])[0]['update_type'] == 'VERSION_UPDATED_ON_OFFICIAL'
        calls = []
        async def fetcher(entry, query):
            calls.append(entry)
            return _slice()
        service = ChatService(LocalSearch(), Model(), official_catalog=[entry], official_fetcher=fetcher)
        result, _ = await _answer(service)
        assert calls == [] and result['retrieved'] == [] and result['citations'] == []
        assert result['official_fallback'][0]['effective_date'] == future
        assert any('日期尚未到' in gap for gap in result['gaps'])
    asyncio.run(run())
