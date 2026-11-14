"""对人工逐项核定的日期补证再次做结构校验，不自动推断法律日期。

阅读入口：review_date。调用者先审阅官方公告及正文版本，再提供明确日期与锚点。
本模块不联网、不写文件；返回新记录和重解析条文，保存与备份由批次执行器负责。
锚点和哈希仅证明所选证据匹配，不能代替现行性或个案适用审查。
"""
from copy import deepcopy
import hashlib
import re
from urllib.parse import urlsplit

from .catalog_ingest import Tree, analyze_document


def review_date(record, raw, evidence_raw, spec, as_of):
    """核验待日期复核记录与官方证据，返回新记录及条文；任一不匹配即拒绝。

    raw/evidence_raw是原始字节，spec包含人工核对的版本锚点、证据锚点和日期。
    不修改输入对象；未来版本仍交给原解析器隔离，不降低正文结构检查门槛。
    """
    if record.get('status') != 'DATE_REVIEW_REQUIRED' or record['title'] != spec['title']:
        raise ValueError('只处理标题匹配的日期隔离记录')
    if hashlib.sha256(raw).hexdigest() != record['raw_sha256']:
        raise ValueError('原文哈希不匹配')
    if hashlib.sha256(evidence_raw).hexdigest() != spec['evidence_sha256']:
        raise ValueError('日期证据哈希不匹配')
    url = urlsplit(spec['evidence_url'])
    if url.scheme != 'https' or url.hostname != 'www.samr.gov.cn' or url.username or url.password:
        raise ValueError('本批次只接受已核查的市场监管总局官方来源')
    html = raw.decode('utf-8')
    root = Tree(html).root
    containers = root.find(cls='Three_xilan_07')
    # 必须绑定法律正文中的修订身份，不能在相关链接里找到年份就认作匹配。
    compact = lambda text: re.sub(r'\s+', '', text)
    if len(containers) != 1 or not spec['body_anchor'] or compact(spec['body_anchor']) not in compact(containers[0].text()):
        raise ValueError('正文版本锚点不匹配')
    if not spec['evidence_anchor'] or compact(spec['evidence_anchor']) not in compact(Tree(evidence_raw.decode('utf-8')).root.text()):
        raise ValueError('官方日期证据锚点不匹配')
    result = deepcopy(record)
    result.update(publication_date=spec['publication_date'], effective_date=spec['effective_date'])
    parsed = analyze_document(result, html, as_of)
    articles = parsed.pop('articles')
    result.update(parsed)
    result['date_review'] = dict(spec, reviewed_as_of=as_of,
                               scope='人工核对版本与施行证据；结构重解析；不认证全部现行性')
    return result, articles
