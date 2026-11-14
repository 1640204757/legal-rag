"""公开官方原件采集：按真实详情元数据取得Word，保留请求、字节和解析记录。

主入口capture_npc（采集）和fetch_official_npc_slice（在线切片）。只写独立暂存，不发布、不生成向量。
已验收隔离正文可只读选条；访问限制立即停止该站点，失败退回目录定位。
URL限定已实际观察的国家库与官网对象存储，不接受用户指定任意地址，不处理验证码或登录。
目录/兜底的纯元数据核验可显式允许官方政府域名；不改变原采集客户端默认主机白名单。
"""
from datetime import date, datetime
import asyncio
import hashlib
import json
from pathlib import Path
import re
import time
import threading
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
import urllib.request
import unicodedata

from .catalog_ingest import _save_json
from .official_texts import read_docx, analyze_plain
from .ingest import ARTICLE, chinese_number

ROOT = Path(__file__).resolve().parents[1]
NPC_FALLBACK_CACHE = ROOT / 'artifacts/local-rag/official-fallback-cache'
NPC_ACCEPTED_STAGE = ROOT / 'data/catalogs/judicial-next-20261022/local-batch-p0-stage1'
_NPC_LOCKS = {}
_NPC_LOCKS_GUARD = threading.Lock()

ALLOWED = {'flk.npc.gov.cn', 'flkoss.obs-bj2.cucloud.cn', 'cicc.court.gov.cn', 'www.court.gov.cn', 'www.npc.gov.cn',
           'fgs.moa.gov.cn', 'policy.mofcom.gov.cn'}


class AccessRestricted(RuntimeError):
    """服务器明确限制访问；调用方必须停止该来源，不能把它当普通超时重试。"""


def public_url(url, *, allow_government=False):
    """返回通过核验的HTTPS地址，失败抛ValueError；仅校验字符串，无网络或写盘副作用。

    默认仍限定既有ALLOWED主机；目录/兜底可显式允许.gov.cn官方域名，不接受伪后缀、
    身份信息、控制字符或非标准端口。该检查不证明正文真实性，抓取器还须核验每次重定向。
    """
    if not isinstance(url, str) or url != url.strip() or any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError('公开来源URL格式无效')
    try:
        parsed = urlsplit(url)
        host, port = parsed.hostname or '', parsed.port
    except ValueError:
        raise ValueError('公开来源URL格式无效') from None
    allowed = host in ALLOWED or (allow_government and host.endswith('.gov.cn'))
    if (parsed.scheme != 'https' or not allowed or parsed.username is not None
            or parsed.password is not None or port not in (None, 443)):
        raise ValueError('正文请求不是已登记的公开官方链路')
    return url


def normalize_catalog_title(title):
    """规范化元数据法名用于同名比较；保留解释序号和试行等限定，无外部副作用。"""
    text = unicodedata.normalize('NFKC', str(title or '')).casefold()
    return re.sub(r'[\s《》〈〉]', '', text)


def catalog_date(value):
    """将明确的年月日或ISO时间转为日期字符串，未知/无效日期返回空；不读取当前日期。"""
    match = re.fullmatch(r'(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?(?:[T ][0-9:.+Z-]+)?', str(value or '').strip())
    if not match:
        return ''
    try:
        return date(*(int(part) for part in match.groups())).isoformat()
    except ValueError:
        return ''


def catalog_entry_metadata(entry):
    """从NPC/行政目录条目返回安全的统一元数据，不可用则返回None；不联网、不改输入。

    目录状态仅为来源声明，不认证个案适用或正文完整；显式不可信URL不得靠bbbs回退掩盖。
    """
    if not isinstance(entry, dict) or not normalize_catalog_title(entry.get('title')):
        return None
    if 'sxx' in entry and str(entry['sxx']).strip() != '3':
        return None
    title = str(entry['title']).strip()
    status = ' '.join(str(entry.get(key) or '') for key in ('status', 'validity', 'state', 'sxx_name', 'sxxName'))
    if any(word in title + status for word in ('征求意见稿', '草案', '废止', '失效')):
        return None
    bbbs = str(entry.get('bbbs') or '').strip()
    urls = [entry[key] for key in ('official_url', 'url', 'source_url') if entry.get(key)]
    if not urls and bbbs:
        urls = ['https://flk.npc.gov.cn/detail?' + urlencode({'id': bbbs})]
    if not urls:
        return None
    try:
        for url in urls:
            public_url(url, allow_government=True)
    except ValueError:
        return None
    publication = catalog_date(entry.get('gbrq') or entry.get('publication_date'))
    effective = catalog_date(entry.get('sxrq') or entry.get('effective_date'))
    result = dict(title=title, official_url=urls[0], url=urls[0], publication_date=publication, effective_date=effective,
                  bbbs=bbbs, gbrq=publication, sxrq=effective, flxz=entry.get('flxz', ''), provider=entry.get('provider', 'npc' if bbbs else 'official'))
    if 'sxx' in entry:
        result['sxx'] = 3
    return result


class PublicClient:
    """低频公开HTTP客户端，原始响应缓存带哈希；不保存Cookie或发送用户咨询内容。"""
    def __init__(self, root):
        self.root = Path(root)
        restriction = self.root/'access_restrictions.json'
        self.denied = set(json.loads(restriction.read_text(encoding='utf-8'))['hosts']) if restriction.exists() else set()
        client = self
        class CheckedRedirect(urllib.request.HTTPRedirectHandler):
            """发出下一跳请求前核对地址与站点限制，而不是响应完成后才发现越界。"""
            def redirect_request(self, request, fp, code, msg, headers, newurl):
                public_url(newurl)
                if urlsplit(newurl).hostname in client.denied:
                    raise AccessRestricted('重定向目标属于已受限来源')
                return super().redirect_request(request, fp, code, msg, headers, newurl)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), CheckedRedirect())

    def restrict(self, host, reason):
        """立即持久化所有访问限制，恢复任务也不能继续请求受限站点。"""
        self.denied.add(host)
        _save_json(self.root/'access_restrictions.json', dict(hosts=sorted(self.denied), reason=reason,
                   recorded_at=datetime.now().astimezone().isoformat()))

    def fetch(self, url, folder, name):
        """取得最大16MB原件，正常断连最多重试一次；401/403/429或验证码不重试。"""
        public_url(url)
        host = urlsplit(url).hostname
        if host in self.denied:
            raise AccessRestricted('该来源已明确拒绝，未继续请求：'+host)
        folder.mkdir(parents=True, exist_ok=True)
        target, meta = folder/name, folder/(name+'.capture.json')
        if target.exists() and meta.exists():
            raw = target.read_bytes()
            capture = json.loads(meta.read_text(encoding='utf-8'))
            if hashlib.sha256(raw).hexdigest() != capture['sha256']:
                raise ValueError('暂存原始响应哈希改变')
            return raw, capture
        for attempt in range(2):
            actual_host = host
            try:
                with self.opener.open(urllib.request.Request(url, headers={'User-Agent': 'LegalRAG-LocalResearch/0.1'}), timeout=35) as response:
                    public_url(response.url)
                    actual_host = urlsplit(response.url).hostname
                    raw = response.read(16_000_001)
                    status = response.status
                if len(raw) > 16_000_000:
                    raise ValueError('原始响应超过大小预算')
                if b'<html' in raw[:500].lower() and any(word in raw.decode('utf-8', 'ignore').lower() for word in ['captcha', '验证码']):
                    raise AccessRestricted('官网返回验证码页面')
                # 临时对象存储签名不写日志；法名、主详情URL与原始文件哈希足以追溯。
                capture = dict(url=url.split('?', 1)[0] if host != 'flk.npc.gov.cn' else url,
                               response_host=actual_host,
                               status=status, retrieved_at=datetime.now().astimezone().isoformat(),
                               bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                target.write_bytes(raw)
                _save_json(meta, capture)
                time.sleep(.5)
                return raw, capture
            except HTTPError as exc:
                if exc.code in {401, 403, 429}:
                    public_url(exc.url)
                    self.restrict(urlsplit(exc.url).hostname, 'HTTP '+str(exc.code))
                    raise AccessRestricted(f'官方来源HTTP {exc.code}，停止该来源') from exc
                if attempt:
                    raise
            except AccessRestricted:
                self.restrict(actual_host, '验证码或来源限制')
                raise
            except (OSError, TimeoutError):
                if attempt:
                    raise
            time.sleep(1)

    def json(self, url, folder, name):
        """记录原始JSON并检查业务状态，失败响应不能冒充可用正文。"""
        raw, capture = self.fetch(url, folder, name)
        value = json.loads(raw)
        if value.get('code') != 200:
            if re.search('验证码|captcha', str(value), re.I):
                self.restrict(capture.get('response_host', urlsplit(url).hostname), '公开接口要求验证码')
                raise AccessRestricted('公开接口要求验证码，停止该来源')
            raise ValueError('官方接口未返回成功状态')
        return value['data'], capture


def capture_npc(entry, client, folder):
    """核对目录与详情标题、日期及时效后下载主DOCX；产出暂存record和articles，不写库。"""
    bbbs = entry['bbbs']
    if not re.fullmatch('[a-fA-F0-9]{32}', bbbs):
        raise ValueError('官方目录版本标识格式异常')
    detail_url = 'https://flk.npc.gov.cn/law-search/search/flfgDetails?'+urlencode({'bbbs': bbbs})
    data, detail_capture = client.json(detail_url, folder, 'detail.json')
    if data.get('bbbs') != bbbs or data.get('title') != entry['title']:
        raise ValueError('详情身份与目录不一致')
    if any(data.get(key) != entry.get(key) for key in ['gbrq', 'sxrq', 'sxx']):
        raise ValueError('目录与在线详情日期/时效已变化，须重新核对版本')
    if data.get('sxx') != 3:
        raise ValueError('官方详情未标记现行有效，隔离')
    if not data.get('ossFile', {}).get('ossWordPath'):
        raise ValueError('官方详情没有主Word原件')
    download_url = 'https://flk.npc.gov.cn/law-search/download/pc?'+urlencode({'format': 'docx', 'bbbs': bbbs, 'fileId': ''})
    download, _ = client.json(download_url, folder, 'download.json')
    raw, capture = client.fetch(download['url'], folder, 'source.docx')
    record = dict(id='catalog-npc-'+bbbs, title=entry['title'], provider='npc',
                  doc_type='JUDICIAL_INTERPRETATION' if entry.get('flxz') == '司法解释' else 'LAW',
                  field='LABOR' if '劳动' in entry['title'] else 'CIVIL' if any(word in entry['title'] for word in ['借贷','买卖合同']) else 'GENERAL',
                  url='https://flk.npc.gov.cn/detail?'+urlencode({'id': bbbs, 'title': entry['title']}),
                  publication_date=data['gbrq'], effective_date=data['sxrq'], raw_file='source.docx',
                  raw_sha256=capture['sha256'], retrieved_at=capture['retrieved_at'],
                  official_catalog_entry=entry, detail_evidence=detail_capture,
                  reuse_permission='UNKNOWN_LOCAL_RETRIEVAL_ONLY_NOT_TRAINING_OR_REDISTRIBUTION')
    parsed = analyze_plain(record, read_docx(raw), date.today().isoformat())
    articles = parsed.pop('articles')
    if not record['publication_date']:
        parsed['status'] = 'DATE_REVIEW_REQUIRED'
    # 解释一存在明确局部废止，现加载器尚不支持款级适用过滤，整体隔离而不是混入旧款。
    if '劳动争议案件适用法律问题的解释（一）' in record['title']:
        parsed['status'] = 'PARTIAL_REPEAL_REVIEW_REQUIRED'
        record['version_issue'] = '2025解释二第二十一条废止解释一第三十二条第一款；款级版本筛选未实现，暂不加载'
        record['version_evidence_url'] = 'https://www.court.gov.cn/zixun/xiangqing/472691.html'
    body = json.dumps(articles, ensure_ascii=False, indent=2).encode('utf-8')
    record.update(parsed, articles=len(articles), parsed_sha256=hashlib.sha256(body).hexdigest())
    (folder/'articles.json').write_bytes(body)
    _save_json(folder/'record.json', record)
    return record


def _npc_detail_identity(url, bbbs):
    """输入详情URL与bbbs，身份一致返回None，否则抛ValueError；无网络或磁盘副作用。"""
    parsed = urlsplit(public_url(url))
    params = parse_qs(parsed.query, keep_blank_values=True)
    identities = [params[key] for key in ('id', 'bbbs') if key in params]
    if parsed.hostname != 'flk.npc.gov.cn' or not identities or any(value != [bbbs] for value in identities):
        raise ValueError('国家库来源与版本身份不一致')


def _regular_npc_files(folder, names):
    """输入目录及文件名，仅查文件元数据，成功无返回值、失败抛ValueError；不创建或修改文件。

    读写前拒绝越界、符号/硬链接和非普通文件；事后哈希不能阻止写穿，原子保存.tmp亦须预检。
    """
    root = folder.resolve()
    for name in names:
        target = folder / name
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError('暂存文件链接或路径越界')
        if target.exists() and (not target.is_file() or target.stat().st_nlink != 1):
            raise ValueError('暂存文件不是独立普通文件')


def _verified_npc_folder(folder, entry, *, acceptance=None):
    """只读核对一份暂存的身份、版本、原件/条文SHA与连续条号，返回记录副本及条文。

    普通缓存只接受PARSED；固定验收暂存可凭匹配的验收条目只读复用隔离正文，不改变准入状态。
    核验失败抛异常供上层安全退回目录定位，不重新写坏缓存或改动data快照。
    """
    _regular_npc_files(folder, ('source.docx', 'articles.json', 'record.json'))
    record = json.loads((folder / 'record.json').read_text(encoding='utf-8'))
    allowed_status = {'PARSED'} if acceptance is None else {'PARSED', 'PARSED_ISOLATED_PENDING_ADMISSION'}
    if record.get('status') not in allowed_status or record.get('raw_file') != 'source.docx':
        raise ValueError('暂存状态或原件格式不可用')
    identities = {record[key] for key in ('npc_bbbs',) if record.get(key)}
    if str(record.get('id', '')).startswith('catalog-npc-'):
        identities.add(record['id'][len('catalog-npc-'):])
    for key in ('official_catalog_entry', 'npc_catalog_entry'):
        metadata = record.get(key) or {}
        if metadata.get('bbbs'):
            identities.add(metadata['bbbs'])
        if 'sxx' in metadata and metadata['sxx'] != 3:
            raise ValueError('暂存来源未标记现行')
    if identities != {entry['bbbs']} or normalize_catalog_title(record.get('title')) != normalize_catalog_title(entry['title']):
        raise ValueError('暂存身份与目录不一致')
    for key in ('publication_date', 'effective_date'):
        value = catalog_date(record.get(key))
        if not value or value != entry[key] or value > date.today().isoformat():
            raise ValueError('暂存日期与当前目录不一致或尚未到期')
    for name in ('source.docx', 'articles.json', 'record.json'):
        if not (folder / name).resolve().is_relative_to(folder.resolve()):
            raise ValueError('暂存文件越界')
    raw, body = (folder / 'source.docx').read_bytes(), (folder / 'articles.json').read_bytes()
    raw_hash, parsed_hash = hashlib.sha256(raw).hexdigest(), hashlib.sha256(body).hexdigest()
    if raw_hash != record.get('raw_sha256') or parsed_hash != record.get('parsed_sha256'):
        raise ValueError('暂存SHA不一致')
    articles = json.loads(body)
    if not isinstance(articles, list) or not articles:
        raise ValueError('没有可用条文')
    base, seen = [], set()
    for article in articles:
        no, text = article['article_no'], article['text']
        heading, body_heading = ARTICLE.match(no + ' '), ARTICLE.match(text)
        if (not heading or not body_heading or heading.group().strip() != no or body_heading.group().strip() != no
                or no in seen or not text[body_heading.end():].strip()):
            raise ValueError('条号重复、错位或正文缺失')
        seen.add(no)
        if not heading[2]:
            base.append(chinese_number(heading[1]))
    if not base or base != list(range(1, base[-1] + 1)) or not re.search('施行|实施|生效|执行', articles[-1]['text']):
        raise ValueError('条号不连续或末条边界不明')
    if record.get('article_count', record.get('articles')) != len(articles):
        raise ValueError('条文数量与记录不一致')
    if acceptance is not None:
        hashes = acceptance.get('hashes', {})
        if (hashes.get('source_docx_sha256') != raw_hash or hashes.get('parsed_articles_sha256') != parsed_hash
                or acceptance.get('article_count') != len(articles)):
            raise ValueError('当前暂存与原验收证据不一致')
    source = record.get('url') or entry['official_url']
    _npc_detail_identity(source, entry['bbbs'])
    return dict(record, url=source), articles


def _accepted_npc_document(entry):
    """按目录条目只读固定验收区，返回(记录,条文)或None；不联网、不写盘，不放行未验收隔离件。"""
    summary_path = NPC_ACCEPTED_STAGE / 'acceptance_summary.json'
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('stage') != 'STAGE1_P0_ISOLATED_ACCEPTANCE':
        return None
    required = ('npc_detail_online', 'title_match', 'npc_sxx_valid', 'court_notice_html_obtained',
                'article_count_exact', 'article_sequence_continuous', 'boundary_first_and_last_valid')
    for result in summary.get('results', []):
        if result.get('npc_bbbs') != entry['bbbs'] or normalize_catalog_title(result.get('title')) != normalize_catalog_title(entry['title']):
            continue
        if not all(result.get('checks', {}).get(key) is True for key in required):
            return None
        candidate_id = result.get('id', '')
        if not isinstance(candidate_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', candidate_id):
            return None
        folder = NPC_ACCEPTED_STAGE / 'documents' / candidate_id
        if not folder.resolve().is_relative_to((NPC_ACCEPTED_STAGE / 'documents').resolve()):
            return None
        return _verified_npc_folder(folder, entry, acceptance=result)
    return None


def _fetch_npc_slice_worker(entry, query, cache_root, client_factory):
    """输入目录/问句/缓存路径/客户端工厂，线程内返回单条字典或None；可能联网及写独立缓存。

    缓存与验收均未命中才创建客户端采集；不保存问题、不发布、不写data或向量库。
    """
    # 同进程相同缓存/版本串行，防止两个请求互相覆盖原件或解析哈希；锁中不保存咨询内容。
    with _NPC_LOCKS_GUARD:
        lock = _NPC_LOCKS.setdefault((str(cache_root), entry['bbbs']), threading.Lock())
    with lock:
        folder = cache_root / entry['bbbs']
        if not folder.resolve().is_relative_to(cache_root):
            return None
        try:
            if (folder / 'record.json').exists():
                # 已登记坏缓存/隔离结果不隐式覆盖，交维护核查；未完成采集且无record才允许续取。
                record, articles = _verified_npc_folder(folder, entry)
            else:
                existing = _accepted_npc_document(entry)
                if existing is not None:
                    record, articles = existing
                else:
                    # 必须在客户端和capture_npc发生任何写入前检查文件级目标，包括原子保存的临时名。
                    names = ('detail.json', 'detail.json.capture.json', 'download.json', 'download.json.capture.json',
                             'source.docx', 'source.docx.capture.json', 'articles.json', 'record.json')
                    _regular_npc_files(folder, names + tuple(Path(name).with_suffix('.tmp').name for name in names))
                    _regular_npc_files(cache_root, ('access_restrictions.json', 'access_restrictions.tmp'))
                    captured = capture_npc(entry, client_factory(cache_root), folder)
                    if captured.get('status') != 'PARSED':
                        return None
                    record, articles = _verified_npc_folder(folder, entry)
            # 延迟导入防止retrieval和官方目录元数据模块形成初始化循环；问题仅用于内存排名。
            from .retrieval import bm25_rank, expand_colloquial_query
            ranked = bm25_rank(articles, expand_colloquial_query(query))
            article = articles[ranked[0][0] if ranked else 0]
            return dict(kind='LAW_ARTICLE', title=record['title'], locator=article['article_no'], text=article['text'],
                        source_url=record['url'], content_scope='ARTICLES_ONLY',
                        version_id=f"{record['publication_date']}（官方白名单实时补充，现行性待核对）")
        except Exception:
            # 网络限制由PublicClient持久化；任何解析/校验失败仅退回目录，不泄露原始响应或咨询。
            return None


async def fetch_official_npc_slice(entry, query, *, cache_root=None, client_factory=PublicClient):
    """返回国家库最相关单条切片或None；先只读验收暂存/缓存，缺失时在线采集到独立缓存。

    entry须含现行国家库bbbs/目录身份和日期，query只用于本机选条，不写盘、不发给国家库。
    cache_root默认artifacts/local-rag/official-fallback-cache，禁止位于data目录；不发布、不写向量库。
    阻塞采集在线程执行；取消可停止本轮等待，但无法强行中断已经发出的线程HTTP请求。
    """
    try:
        metadata = catalog_entry_metadata(entry)
        if (metadata is None or metadata.get('sxx') != 3 or not re.fullmatch(r'[a-fA-F0-9]{32}', metadata['bbbs'])
                or not metadata['publication_date'] or not metadata['effective_date']):
            return None
        _npc_detail_identity(metadata['official_url'], metadata['bbbs'])
        if any(metadata[key] > date.today().isoformat() for key in ('publication_date', 'effective_date')):
            return None
        cache = Path(cache_root if cache_root is not None else NPC_FALLBACK_CACHE).resolve()
        if cache.is_relative_to((ROOT / 'data').resolve()):
            return None
    except (ValueError, TypeError, OSError):
        return None
    return await asyncio.to_thread(_fetch_npc_slice_worker, metadata, query, cache, client_factory)
