"""国家库兜底落地回归：缓存/验收暂存只读复用、受控采集、口语专题与网页接入。

采集和模型用替身隔离网络；DOM测试直接执行生产app.js，不创建实验脚本或修改法规快照。
"""
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading

import pytest
from fastapi.testclient import TestClient

import local_rag.app as app_module
import local_rag.official_capture as capture
import local_rag.retrieval as retrieval
from local_rag.service import ChatService


ROOT = Path(__file__).resolve().parents[1]
ENTRY = dict(bbbs='a' * 32, title='最高人民法院关于人脸识别的测试规定', gbrq='2021-07-27', sxrq='2021-08-01', sxx=3,
             official_url='https://flk.npc.gov.cn/detail?id=' + 'a' * 32, flxz='司法解释')
ARTICLES = [dict(article_no='第一条', number=1, inserted=False, text='第一条 本规定保护合法权益。'),
            dict(article_no='第二条', number=2, inserted=False,
                 text='第二条 物业服务企业不得强制刷脸，应当提供合理验证方式。本规定自2021年8月1日起施行。')]


def _seed(folder, entry=ENTRY, status='PARSED', articles=None):
    """只在tmp_path创建完整缓存替身，返回记录；原件字节仅用于哈希，不冒充真实DOCX。"""
    folder.mkdir(parents=True, exist_ok=True)
    raw = b'test official docx bytes'
    body = json.dumps(articles or ARTICLES, ensure_ascii=False).encode('utf-8')
    record = dict(id='catalog-npc-' + entry['bbbs'], title=entry['title'], status=status, raw_file='source.docx',
                  publication_date=entry['gbrq'], effective_date=entry['sxrq'], url=entry['official_url'],
                  raw_sha256=hashlib.sha256(raw).hexdigest(), parsed_sha256=hashlib.sha256(body).hexdigest(),
                  articles=len(articles or ARTICLES), official_catalog_entry=entry)
    (folder / 'source.docx').write_bytes(raw)
    (folder / 'articles.json').write_bytes(body)
    (folder / 'record.json').write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    return record


@pytest.fixture(autouse=True)
def isolate_accepted_stage(tmp_path, monkeypatch):
    """合成测试不能意外命中用户真实暂存；专门的真实暂存测试会显式恢复路径。"""
    monkeypatch.setattr(capture, 'NPC_ACCEPTED_STAGE', tmp_path / 'accepted', raising=False)


@pytest.mark.parametrize('query, article', [('物业不让我进小区，非要刷脸怎么办', '第二条'), ('zzzzunrelated', '第一条')])
def test_verified_cache_skips_client_and_selects_relevant_article(tmp_path, query, article):
    """完整哈希缓存直接读取，词面选择相关条款，无重合时回退第一条；不得创建网络客户端。"""
    cache = tmp_path / 'cache'
    _seed(cache / ENTRY['bbbs'])
    def forbidden(_):
        raise AssertionError('缓存命中不应创建网络客户端')
    row = asyncio.run(capture.fetch_official_npc_slice(ENTRY, query, cache_root=cache, client_factory=forbidden))
    assert row and row['locator'] == article
    assert row['source_url'] == ENTRY['official_url'] and row['title'] == ENTRY['title']
    assert row['version_id'] == '2021-07-27（官方白名单实时补充，现行性待核对）'


def test_cache_miss_captures_on_worker_thread_and_reuses_result(tmp_path, monkeypatch):
    """未缓存时调用实际采集入口的替身，线程内工作且后续查询不重复下载。"""
    calls = []
    main_thread = threading.get_ident()
    def fake_capture(entry, client, folder):
        calls.append((entry, client, threading.get_ident()))
        return _seed(folder, entry)
    monkeypatch.setattr(capture, 'capture_npc', fake_capture)
    cache = tmp_path / 'cache'
    row = asyncio.run(capture.fetch_official_npc_slice(ENTRY, '强制刷脸', cache_root=cache, client_factory=lambda root: 'client'))
    again = asyncio.run(capture.fetch_official_npc_slice(ENTRY, '强制刷脸', cache_root=cache, client_factory=lambda root: 'unused'))
    assert row['locator'] == again['locator'] == '第二条'
    assert len(calls) == 1 and calls[0][1] == 'client' and calls[0][2] != main_thread


@pytest.mark.parametrize('corruption', ['status', 'raw_hash', 'parsed_hash', 'title', 'identity', 'article_gap', 'source_identity',
                                      'source_empty_id', 'source_empty_bbbs'])
def test_existing_invalid_cache_is_not_used_or_silently_overwritten(tmp_path, monkeypatch, corruption):
    """坏缓存、隔离状态或身份不符不放行，也不联网覆盖已有证据。"""
    folder = tmp_path / 'cache' / ENTRY['bbbs']
    record = _seed(folder)
    if corruption == 'status':
        record['status'] = 'QUARANTINED'
    elif corruption == 'raw_hash':
        (folder / 'source.docx').write_bytes(b'changed')
    elif corruption == 'parsed_hash':
        (folder / 'articles.json').write_bytes(b'[]')
    elif corruption == 'title':
        record['title'] = '另一部法规'
    elif corruption == 'identity':
        record['official_catalog_entry'] = dict(ENTRY, bbbs='b' * 32)
    elif corruption == 'source_identity':
        record['url'] = 'https://flk.npc.gov.cn/detail?id=' + 'b' * 32
    elif corruption in ('source_empty_id', 'source_empty_bbbs'):
        record['url'] = ENTRY['official_url'] + ('&id=' if corruption == 'source_empty_id' else '&bbbs=')
    elif corruption == 'article_gap':
        body = json.dumps([ARTICLES[0], dict(ARTICLES[1], article_no='第三条', number=3,
                                           text=ARTICLES[1]['text'].replace('第二条', '第三条'))], ensure_ascii=False).encode('utf-8')
        (folder / 'articles.json').write_bytes(body)
        record['parsed_sha256'] = hashlib.sha256(body).hexdigest()
    (folder / 'record.json').write_text(json.dumps(record), encoding='utf-8')
    def forbidden(*args):
        raise AssertionError('无效缓存不能触发自动覆盖')
    monkeypatch.setattr(capture, 'capture_npc', forbidden)
    assert asyncio.run(capture.fetch_official_npc_slice(ENTRY, '刷脸', cache_root=folder.parent, client_factory=forbidden)) is None


@pytest.mark.parametrize('changes', [dict(official_url='https://evil.test/law'), dict(bbbs='../escape'), dict(sxx=2),
                                     dict(official_url='https://flk.npc.gov.cn/detail?id=' + 'b' * 32),
                                     dict(official_url=ENTRY['official_url'] + '&id='),
                                     dict(official_url=ENTRY['official_url'] + '&bbbs=')])
def test_npc_fetch_rejects_untrusted_entry_before_side_effects(tmp_path, changes):
    """仅允许国家库版本详情，拒绝非官方来源、路径穿越、失效或详情身份不符。"""
    calls = []
    row = asyncio.run(capture.fetch_official_npc_slice(dict(ENTRY, **changes), '刷脸', cache_root=tmp_path / 'cache',
                                                      client_factory=lambda root: calls.append(root)))
    assert row is None and calls == [] and not (tmp_path / 'cache').exists()


def test_capture_quarantine_returns_none(tmp_path, monkeypatch):
    """实时解析状态非PARSED时不选条文，不把隔离文件冒充可引用材料。"""
    monkeypatch.setattr(capture, 'capture_npc', lambda entry, client, folder: _seed(folder, entry, 'DATE_REVIEW_REQUIRED'))
    assert asyncio.run(capture.fetch_official_npc_slice(ENTRY, '刷脸', cache_root=tmp_path / 'cache', client_factory=lambda root: object())) is None


@pytest.mark.parametrize('name', ['source.docx', 'articles.json', 'detail.json', 'access_restrictions.json',
                                'detail.json.capture.tmp', 'source.docx.capture.tmp', 'record.tmp', 'access_restrictions.tmp'])
def test_linked_cache_targets_are_rejected_before_any_client_or_capture_write(tmp_path, monkeypatch, name):
    """真实硬链接模拟缓存文件别名：写入前必须拒绝，不能先污染外部文件再做哈希检查。"""
    cache = tmp_path / 'cache'
    folder = cache / ENTRY['bbbs']
    folder.mkdir(parents=True)
    victim = tmp_path / 'protected-original.json'
    original = b'{"hosts":[]}'
    victim.write_bytes(original)
    linked = (cache if name.startswith('access_restrictions') else folder) / name
    linked.hardlink_to(victim)
    calls = []
    def client_factory(root):
        calls.append('client')
        if name.startswith('access_restrictions'):
            if name.endswith('.tmp'):
                capture._save_json(root / 'access_restrictions.json', {'hosts': ['changed']})
            else:
                (root / name).write_bytes(b'changed')
        return object()
    def fake_capture(entry, client, target):
        calls.append('capture')
        if name == 'detail.json':
            (target / name).write_bytes(b'changed')
        elif name.endswith('.tmp') and not name.startswith('access_restrictions'):
            capture._save_json(target / (name[:-4] + '.json'), {'changed': True})
        return _seed(target, entry)
    monkeypatch.setattr(capture, 'capture_npc', fake_capture)
    result = asyncio.run(capture.fetch_official_npc_slice(ENTRY, '刷脸', cache_root=cache, client_factory=client_factory))
    assert result is None and calls == []
    assert victim.read_bytes() == original


def test_live_cache_cannot_be_configured_under_original_data():
    """配置错误也不能把运行缓存写到data目录；校验必须在创建客户端之前完成。"""
    calls = []
    result = asyncio.run(capture.fetch_official_npc_slice(ENTRY, '刷脸', cache_root=ROOT / 'data/library',
                                                        client_factory=lambda root: calls.append(root)))
    assert result is None and calls == []


def test_real_accepted_isolated_stage_is_read_only_and_avoids_network(tmp_path, monkeypatch):
    """只读真实验收汇总允许的暂存，不改变隔离准入状态或写入默认缓存。"""
    stage = ROOT / 'data/catalogs/judicial-next-20261022/local-batch-p0-stage1'
    monkeypatch.setattr(capture, 'NPC_ACCEPTED_STAGE', stage)
    record_path = stage / 'documents/J18/record.json'
    before = record_path.read_bytes()
    record = json.loads(before)
    entry = record['npc_catalog_entry']
    entry = dict(entry, official_url='https://flk.npc.gov.cn/detail?id=' + entry['bbbs'])
    def forbidden(_):
        raise AssertionError('已验收暂存命中不应创建客户端')
    row = asyncio.run(capture.fetch_official_npc_slice(entry, '申请财产保全需要提交哪些财产线索和申请书',
                                                      cache_root=tmp_path / 'cache', client_factory=forbidden))
    assert row and '申请书' in row['text']
    assert record_path.read_bytes() == before and not (tmp_path / 'cache').exists()


@pytest.mark.parametrize('query, title_part', [
    ('小区进门物业非要刷脸，不录入人脸就不给进怎么办', '使用人脸识别技术处理个人信息'),
    ('人脸识别怎么办', '使用人脸识别技术处理个人信息'),
    ('诈骗刑事立案标准是什么', '办理诈骗刑事案件具体应用法律'),
    ('网购商品拆封后商家说拆封不退怎么办', '审理网络消费纠纷'),
    ('超市买到过期食品能要求退一赔十吗', '审理食品药品惩罚性赔偿'),
    ('买期房逾期交房不办证怎么办', '商品房买卖合同'),
    ('微信聊天记录怎么当证据', '民事诉讼证据'),
    ('被骗五千块钱够诈骗立案标准吗', '办理诈骗刑事案件具体应用法律'),
    ('被骗五千块够刑事立案吗', '办理诈骗刑事案件具体应用法律'),
    ('钱包被偷走了多少钱立案', '办理盗窃刑事案件适用法律'),
])
def test_colloquial_topics_match_real_judicial_catalog(query, title_part):
    """无正式专题长词的口语应能凭明确扩展命中真实目录，不改官方标题。"""
    catalog = json.loads((ROOT / 'data/catalogs/official-20261020/npc/judicial/catalog.json').read_text(encoding='utf-8'))['entries']
    matches = retrieval.match_official_whitelist_catalog(query, catalog, set())
    assert matches and any(title_part in entry['title'] for entry in matches), matches
    assert all(entry['official_url'].startswith('https://flk.npc.gov.cn/detail?id=') for entry in matches)


@pytest.mark.parametrize('phrase, term', [
    *[(p, '使用人脸识别技术处理个人信息') for p in ('刷脸', '人脸识别')],
    *[(p, '审理网络消费纠纷') for p in ('拆封不退', '七天无理由', '直播带货', '网购')],
    *[(p, '审理食品药品惩罚性赔偿') for p in ('退一赔十', '过期食品', '变质食品', '假药')],
    *[(p, '审理商品房买卖合同') for p in ('逾期交房', '延期交房', '逾期办证', '一房二卖', '买期房')],
    *[(p, '办理财产保全案件') for p in ('财产保全', '冻结账户', '查封财产')],
    *[(p, '民事诉讼证据') for p in ('微信聊天记录', '电子证据', '录音证据')],
    *[(p, '办理诈骗刑事案件具体应用法律') for p in ('立案标准', '多少钱立案', '够判刑吗', '够刑事立案吗')],
])
def test_each_colloquial_trigger_supplies_catalog_topic(phrase, term):
    """逐个触发词补专题，重复命中的完整词串不重复追加。"""
    query = phrase + '怎么办'
    expanded = retrieval.expand_colloquial_query(query)
    assert expanded.startswith(query + ' ') and expanded[len(query):].split().count(term) == 1


@pytest.mark.parametrize('query', ['强奸案件立案标准是什么', '故意伤害多少钱立案', '立案标准是什么'])
def test_generic_filing_language_does_not_invent_fraud_or_theft_context(query):
    """明确其他罪名或未交代犯罪类型时，不能凭立案模板扩展触发诈骗/盗窃目录。"""
    catalog = json.loads((ROOT / 'data/catalogs/official-20261020/npc/judicial/catalog.json').read_text(encoding='utf-8'))['entries']
    matches = retrieval.match_official_whitelist_catalog(query, catalog, set())
    assert not any('诈骗刑事' in entry['title'] or '盗窃刑事' in entry['title'] for entry in matches), matches


def test_unrelated_crime_does_not_call_official_fetcher():
    """其他刑事场景即使提及立案，也不得在服务层调用诈骗/盗窃抓取器。"""
    async def run():
        catalog = json.loads((ROOT / 'data/catalogs/official-20261020/npc/judicial/catalog.json').read_text(encoding='utf-8'))['entries']
        catalog = [entry for entry in catalog if '诈骗刑事' in entry['title'] or '盗窃刑事' in entry['title']]
        class Search:
            rows = [dict(field='CRIMINAL', title='中华人民共和国刑法')]
            async def search(self, query):
                return [], 'lexical'
        class Model:
            async def chat(self, messages):
                return '## 追问\n请补充事实。\n## 处理方向\n核实情况。\n## 证据清单\n保留证据。\n## 不确定性\n材料不足。\nUSED_IDS: 无'
        calls = []
        async def fetcher(entry, query):
            calls.append(entry)
        service = ChatService(Search(), Model(), official_catalog=catalog, official_fetcher=fetcher)
        sid = service.create()
        rid = service.start(sid, '强奸案件立案标准是什么')
        await service.sessions[sid].jobs[rid]['task']
        job = service.result(sid, rid)
        assert job['status'] == 'done', job
        assert calls == [] and job['result']['official_fallback'] == []
    asyncio.run(run())


def test_default_web_binds_real_fetcher_but_direct_service_stays_offline(monkeypatch):
    """只在默认Web装配启用真实抓取器；测试注入服务及直接构造保持离线。"""
    original = app_module.ChatService
    created = []
    def factory(retriever, model, **kwargs):
        service = original(retriever, model, official_catalog=[], **kwargs)
        created.append(service)
        return service
    class Model:
        def __init__(self, *args):
            pass
        async def status(self):
            return {'embedding_ready': False}
        async def close(self):
            pass
    class Store:
        ready = False
        async def close(self):
            pass
    monkeypatch.setattr(app_module, 'ChatService', factory)
    monkeypatch.setattr(app_module, 'OllamaClient', Model)
    monkeypatch.setattr(app_module, 'QdrantStore', lambda *args: Store())
    monkeypatch.setattr(app_module, 'load_corpus', lambda: ([], {}))
    with TestClient(app_module.create_app()):
        assert len(created) == 1 and created[0].official_fetcher is capture.fetch_official_npc_slice
    offline = original(object(), object(), official_catalog=[])
    assert offline.official_fetcher is None
    with TestClient(app_module.create_app(offline)):
        assert offline.official_fetcher is None and len(created) == 1


_DOM_RUNNER = r'''
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this._text = ''; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set innerHTML(value) { throw new Error('禁止HTML字符串注入'); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this._text = ''; this.children = items; }
  addEventListener() {}
}
const elements = new Map();
const ctx = vm.createContext({ URL, input, document: {
  querySelectorAll() { return []; },
  getElementById(id) { if (!elements.has(id)) elements.set(id, new Element('div')); return elements.get(id); },
  createElement(tag) { return new Element(tag); }, createTextNode(s) { const n=new Element('text'); n.textContent=s; return n; }
}, window: { addEventListener() {} }, fetch() { return new Promise(() => {}); }, setInterval() {} });
vm.runInContext(fs.readFileSync(input.path, 'utf8'), ctx);
vm.runInContext('lastResult = input.result; showAll = input.showAll; sources();', ctx);
function plain(e) { return { tag:e.tag, text:e.textContent, href:e.href || null, children:e.children.map(plain) }; }
process.stdout.write(JSON.stringify(plain(elements.get('sources'))));
'''


def _render_sources(result, show_all):
    """在最小DOM替身执行生产JS；初始化fetch悬置，不联网、不写文件。"""
    node = shutil.which('node')
    if not node:
        pytest.skip('需要Node执行生产来源面板DOM回归')
    payload = dict(path=str(ROOT / 'local_rag/static/app.js'), result=result, showAll=show_all)
    run = subprocess.run([node, '-e', _DOM_RUNNER], input=json.dumps(payload), text=True, encoding='utf-8',
                         capture_output=True, timeout=15)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def test_source_panel_labels_live_slices_and_uses_text_nodes():
    """实时卡片显示专用标签，恶意HTML标题仍仅是文字而不是DOM。"""
    row = dict(material_id='M1', title='<img src=x onerror=alert(1)>', text='第一条 正文。', locator='第一条',
               content_scope='ARTICLES_ONLY', verified_scope='官方白名单实时检索补充；已核验来源', source_url=ENTRY['official_url'])
    rendered = _render_sources(dict(citations=[row], retrieved=[row], official_fallback=[]), False)
    assert '官方白名单实时补充 · 未核实个案适用' in rendered['text']
    assert '未逐条在线复核是否仍然有效' in rendered['text']
    assert '版本待核实' in rendered['text']
    assert '<img src=x' in rendered['text']
    assert not any(child['tag'] == 'img' for child in rendered['children'])


def test_source_panel_shows_captured_version_on_every_card():
    """已用引用、前8条和更多候选都写出采集版本，不能只在页顶声明一次。"""
    row = dict(material_id='M1', title='测试规范', text='第一条 正文。', locator='第一条',
               version_id='2020-05-28', content_scope='FULL_TEXT', verified_scope='与已保存原文快照逐字一致',
               source_url='https://www.gov.cn/test')
    tail = dict(row, material_id=None, sent_to_model=False, routes={'bm25': 12, 'dense': 4})
    rendered = _render_sources(dict(
        citations=[row], retrieved=[row], display=dict(top=[dict(row, sent_to_model=True)], tail=[tail]),
        official_fallback=[]), True)
    assert rendered['text'].count('采集版本 2020-05-28') >= 2
    assert '未核个案适用' in rendered['text']


@pytest.mark.parametrize('url, linked', [(ENTRY['official_url'], True), ('javascript:alert(1)', False), ('https://evil.test/law', False)])
def test_candidate_panel_shows_directory_even_without_article_rows(url, linked):
    """无条文时也显示未送模目录提示，来源链接须是安全官方HTTPS，标题只按文本展示。"""
    entry = dict(title='<script>坏标题</script>', effective_date='2021-08-01', official_url=url, sent_to_model=False)
    result = dict(citations=[], retrieved=[], official_fallback=[entry])
    rendered = _render_sources(result, True)
    assert '<script>坏标题</script>' in rendered['text'] and '2021-08-01' in rendered['text']
    assert '未送入模型' in rendered['text']
    cards = [child for child in rendered['children'] if child['tag'] == 'article']
    links = [child for card in cards for child in card['children'] if child['tag'] == 'a']
    assert bool(links) == linked
    if linked:
        assert links[0]['text'] == '查看官方目录来源'
    cited_view = _render_sources(result, False)
    assert '<script>坏标题</script>' not in cited_view['text']
